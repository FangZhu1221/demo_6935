# Final project of Bigdata Eco-system
### README

Team 25, Insaiyan

### Steps

[1]. Follow the README.md file in the folder Speech2Text.

```
cd ./Speech2Text
```

[2]. Install the requirements in the folder Speech2Text.

```
sudo pip3 install -r requirements.txt
```

[3]. Translate the voice into text message.

```
sudo python3 trancribe.py --model-path models/librispeech_final2.pth
```

[3]. Reorder the text message

```
sudo python3 transcribelistorder.py
```

[4]. Enter the folder of VDCNN

```
cd ../VDCNN
```

[6]. Run the VDCNN to get the accuracy.

```
sudo python3 test1.py
```

