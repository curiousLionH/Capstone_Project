import serial
import time

ser = serial.Serial(
    # port='/dev/cu.HC-06-DevB',
    port= 'COM1',    # Port Number
    baudrate=9600,
)

A = "12"
B = "34"
C = "56"

Trans = "Q" + A + B + C
print(f"sending data {Trans} ...")
Trans = Trans.encode("utf-8")

starttime = time.time()

while (time.time() - starttime) <= 2:
    ser.write(Trans)