# Pytorch_OCR_HandWritenMathematicalRecognition_Project

模型一：
![image](https://github.com/user-attachments/assets/5af6595b-c52e-4831-a3c7-058aebce896c)
把词典，测试数据集，训练数据集的路径给改掉就可以直接跑了。

![image](https://github.com/user-attachments/assets/d5899a13-38a2-4263-96a7-cf4b14fdb530)
如果要继续训练就写上.pth文件的路径

dataloader.py中：
![image](https://github.com/user-attachments/assets/31f18953-5c05-472e-a636-3e6d76e3b24c)
把注释取消，return 2给注释了。设置为2为了方便调试。
训练:python train.py
测试:python test.py


模型二：
先修改config文件：
![image](https://github.com/user-attachments/assets/3600f3c7-dfd9-46f0-b243-4714da606cee)
![image](https://github.com/user-attachments/assets/e1464540-8c6d-44c5-a9b9-1b2242ca5248)
带mnt的修改掉即可。
训练：python train.py
测试: python inference.py
