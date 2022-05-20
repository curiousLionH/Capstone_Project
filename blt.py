import serial

ser = serial.Serial(
    # port='/dev/cu.HC-06-DevB',
    port='/dev/ttyACM1',
    baudrate=9600,
)

while True:
    temp_list = []
    while ser.readable():
        a = ser.read().decode()
        if (a == '\n'):
            break
        temp_list.append(a)
    print(temp_list)
    try:
        if (temp_list[0] == 'A'):
            dfjldsfj = int(temp_list[1])
            print(temp_list[1])
    except:
        pass