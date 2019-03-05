set root=A:\Anaconda3

call %root%\Scripts\activate.bat %root%

call tensorboard --logdir=best_logs/ --host localhost --port 8080