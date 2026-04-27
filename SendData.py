import serial
import serial.tools.list_ports as ser

serial_object = serial.Serial(port = "COM3", baudrate = 9600)

while True:
    x = input("Quit (Q)? \n")

    if (x == 'Q'):
        break
    
    angles = []
    for i in range(1,7):
        angle = input(f'Enter angle for motor {i} \n' )
        angles.append(angle)

    print(angles)
    angles_str = ','.join(map(str,angles)) +',\n'
    print(angles_str)

    serial_object.write(angles_str.encode('utf-8'))
 
