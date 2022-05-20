import serial
import time

# ser = serial.Serial(
#     # port='/dev/cu.HC-06-DevB',
#     port="COM3",  # Port Number
#     baudrate=9600,
# )

ser = serial.Serial("COM3", 9600)

A = "34"
B = "56"
C = "78"

Trans = "Q" + A + B + C

Trans = Trans.encode("utf-8")
print(f"sending data {Trans} ...")

starttime = time.time()

while True:
    ser.write(Trans)
    time.sleep(1)
