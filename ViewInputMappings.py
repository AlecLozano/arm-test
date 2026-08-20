import pygame
import time
import os
from platform import system
from ControllerCommon.XboxController import XboxController

def main():
    """Connects to any detected Controller and gives a live display of all it's inputs that are detected by Pygame.
    The ID's of the axis/buttons/D-pads will map to any getRawInput function in Controller Common"""

    print("bleh :p")

    connected = False
    while not connected:
        try:
            pygame.init()
            controller = XboxController(deadZone=0.5)
            connected = True
        except pygame.error:
            print("Couldn't connect to controller")
            pygame.joystick.quit()
            time.sleep(1)

    if (system() == "Windows"):
        os.system('cls')
    else:
        os.system('clear')

    
    while True:

        inputs = controller.getControllerInput()
        for i in inputs:
            id = controller.getInputID(i)
            val = controller.getInputValue(i)
            print(f"{id}:{val}")


        

if __name__ == "__main__":
    main()