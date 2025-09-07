on client side :
    copy *.bin to each client

on server side : 
    sudo docker exec -ti otbr1 sh
    cd Agregation_FL
    cd aiocoap
    python3.7 multimfloat16.py

Results:

with: 9 client;batch = 330;SGDit =15(14 counting 0); FLit = 120

Training accuracy =   0.9539388
Evaluation accuracy = 0.811111
comunication in   = 12 441 600
comunication out  =  1 393 920
comunication cost = 13 835 520