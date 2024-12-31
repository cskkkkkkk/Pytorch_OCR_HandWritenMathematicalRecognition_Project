import time
from config import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from model.utils import *
from model import metrics,dataloader,model
from torch.utils.checkpoint import checkpoint as train_ck
from torch.utils.data import DataLoader

from model.dataloader import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model.device = device
'''
如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
'''
# cudnn.benchmark = True


def main():
    """
    Training and validation.
    """

    global best_score, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # 字典文件
    # word_map = load_json(vocab_path)
    with open(vocab_path) as f:
        words = f.readlines()
    words.append("<start>")
    words.append("<end>")
    # vocab = load_json(vocab_path)
    word_map = {value: index + 1 for index, value in enumerate(words)}
    word_map["<pad>"] = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = model.DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = model.Encoder()
        # encoder_optimizer = None
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_score = checkpoint['score']
        decoder = checkpoint['decoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        # encoder_optimizer = checkpoint['encoder_optimizer']
        # encoder_optimizer = None
        # if fine_tune_encoder is True and encoder_optimizer is None:
        #     encoder.fine_tune(fine_tune_encoder)
        #     encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
        #                                          lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # test_params_flop(encoder, (1, 64, 64))
    # test_params_flop(decoder, (512, 32, 32))
    # exit(1)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 自定义的数据集
    train_dataset = MyDataset(dataset_dir, is_train=True)
    eval_dataset = MyDataset(dataset_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=0
    )
    val_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=0
    )
    # 动态学习率调度器
    scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # #统计验证集的词频
    # words_freq = cal_word_freq(word_map,val_loader)
    # print(words_freq)
    p = 1#teacher forcing概率
    # Epochs
    for epoch in tqdm(range(start_epoch, epochs)):
        #每2个epoch衰减一次teahcer forcing的概率
        if p > 0.05:
            if (epoch % 3 == 0 and epoch != 0):
                p *= 0.75
        else:
            p = 0

        # 如果迭代4次后没有改善,则对学习率进行衰减,如果迭代20次都没有改善则触发早停.直到最大迭代次数
        if epochs_since_improvement == 30:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            adjust_learning_rate(encoder_optimizer, 0.8)
        #动态学习率调节
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, 
        #     patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=decoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,p=p)#encoder_optimizer=encoder_optimizer,

        # One epoch's validation
        recent_score = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        print("p:",p)
        if (p==0):
            print('Stop teacher forcing!')
            # Check if there was an improvement
            is_best = recent_score > best_score
            best_score = max(recent_score, best_score)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                print('New Best Score!(%d)'%(best_score,))
                epochs_since_improvement = 0

            if epoch % save_freq == 0:
                print('Saveing...')
                save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,encoder_optimizer,
                            decoder_optimizer, recent_score, is_best)
                             # 更新学习率调度器
            scheduler_decoder.step(recent_score)
            scheduler_encoder.step(recent_score)
        print('--------------------------------------------------------------------------')


def train(train_loader, encoder, decoder, criterion, encoder_optimizer,decoder_optimizer, epoch, p):
    """
    Performs one epoch's training.
    :param train_loader: 训练集的dataloader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    # Batches
    #tqdm 函数接受一个可迭代对象，并返回一个新的迭代器，该迭代器会在遍历过程中显示一个进度条。
    with tqdm(enumerate(train_loader), total=len(train_loader)) as it:
        for i, (imgs, caps, caplens) in it:
            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            #print("---------------------------------------------------------")
            #print("imgs:",imgs.shape)
            try:
                imgs = encoder(imgs)
            except:
                imgs = train_ck(encoder,imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, p=p)

            # 由于加入开始符<start>以及停止符<end>,caption从第二位开始,知道结束符
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = pack_padded_sequence(scores, decode_lengths.cpu().int(), batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths.cpu().int(), batch_first=True).data

            # Calculate loss
            scores = scores.to(device)
            loss = criterion(scores, targets)

            # 加入 doubly stochastic attention 正则化
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # 反向传播
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                # if encoder_optimizer is not None:
                #     clip_gradient(encoder_optimizer, grad_clip)

            # 更新权重
            decoder_optimizer.step()
            encoder_optimizer.step()
            # if encoder_optimizer is not None:
            #     encoder_optimizer.step()

            it.set_postfix(
                Loss=f"{loss:.2e}",
            )
            
            # if i % save_freq == 0:
            #     save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,encoder_optimizer,
            #                 decoder_optimizer, 0,0)


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: 用于验证集的dataloader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :return: 验证集上的BLEU-4 score
    """
    decoder.eval()  # 推断模式,取消dropout以及批标准化
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top3accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        with tqdm(enumerate(val_loader), leave=False, total=len(val_loader)) as it:
            for i, (imgs, caps, caplens) in it:

                # Move to device, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                # Forward prop.
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, p=0)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths.cpu().int(), batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths.cpu().int(), batch_first=True).data

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top3 = accuracy(scores, targets, 3)
                top3accs.update(top3, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                it.set_postfix(
                    Loss=f"{loss:.2e}",
                )

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

                # References
                # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
                # for j in range(allcaps.shape[0]):
                #     img_caps = allcaps[j].tolist()
                #     img_captions = list(
                #         map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                #             img_caps))  # remove <start> and pads
                #     references.append(img_captions)
                caplens = caplens[sort_ind]
                caps = caps[sort_ind]
                for i in range(len(caplens)):
                    caplen = int(caplens[i].item())
                    references.append(caps[i][1:caplen].tolist())
                # Hypotheses
                # 这里直接使用greedy模式进行评价,在推断中一般使用集束搜索模式
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    decode_len=int(decode_lengths[j].item())
                    temp_preds.append(preds[j][:decode_len])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

            Score = metrics.evaluate(losses, top3accs, references, hypotheses)
    return Score

def test_model():   
    with open(vocab_path) as f:
        words = f.readlines()
    words.append("<start>")
    words.append("<end>")
    # vocab = load_json(vocab_path)
    word_map = {value: index + 1 for index, value in enumerate(words)}
    word_map["<pad>"] = 0
    # 初始化模型和优化器
    encoder = model.Encoder()        
    decoder = model.DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
    # 加载最佳模型参数文件
    checkpoint_path = './BEST_checkpoint_MyDataset.pth'
    checkpoint = torch.load(checkpoint_path)
    encoder_optimizer = checkpoint['encoder_optimizer']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    #
    epoch=checkpoint['epoch']
    epochs_since_improvement=checkpoint['epochs_since_improvement']
    best_score= checkpoint['score']
    
    

    print("encoder:",test_params_flop(encoder, (1, 64, 64)))
    print("decoder:",test_params_flop(decoder, (512, 32, 32)))
    

if __name__ == '__main__':
    main()
    test_model()
