import socket
import csv

local_ip = "0.0.0.0"
# local_ip = "192.168.137.222"
local_port = 4210
filename = "new_train_set10.csv"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((local_ip, local_port))

print("listening to port " + str(local_port))

with open(filename, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])

    try:
        while True:
            # buffer size 1024 bytes
            data, addr = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            values = line.split(',')
            print("received")
            writer.writerow(values)
    except KeyboardInterrupt:
        print("transmission stopped")
