on client side :
    copy *.bin to each client

on server side : 
    sudo docker exec -ti otbr1 sh
    cd Agregation_FL
    cd aiocoap
    python3.7 multim.py


Results:

with: 9 client;batch = 330;SGDit =15(14 counting 0); FLit = 120

Training accuracy =   0.9511784
Evaluation accuracy = 0.8259259
comunication in   = 20 736 000
comunication out  =  2 323 200
comunication cost = 23 059 200