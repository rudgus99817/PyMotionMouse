from pyautogui import _pyautogui_win as mouse


class Control:
    def __init__(self):
        self.current_pos = mouse._position()
        self.clicked = False
        x, y = mouse._size()
        x, y = int(x/2), int(y/2)
        mouse._moveTo(x, y)

    def move(self, prior, current):
        moved = (current[0] - prior[0], current[1] - prior[1])
        if(abs(moved[0])<5 and abs(moved[1])<5):
            return
        moved = (moved[0]*3, moved[1]*5)
        pos = mouse._position()
        self.current_pos = (pos[0]-moved[0], pos[1] + moved[1])
        mouse._moveTo(self.current_pos[0], self.current_pos[1])

    def clickstat(self, clicked):
        if(self.clicked == True and clicked == False):
            self.clicked = clicked
            return True
        else:
            self.clicked = clicked
            return False

    def click(self):
        mouse._click(self.current_pos[0], self.current_pos[1], "left")

    def double_click(self):
        mouse._click(self.current_pos[0], self.current_pos[1], "left")
        mouse._click(self.current_pos[0], self.current_pos[1], "left")