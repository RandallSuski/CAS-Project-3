import tkinter as tk

class CellularAutomata2D(tk.Tk):
    def __init__(self, width, height):
        super().__init__() #Initialize the tkinter class 
        self.width = width
        self.height = height

    def create_time_space_diagram(self, ca_time_space, length):
        # Create a background canvas
        canvas = tk.Canvas(self, width=self.width, height=self.height, bg='white')
        canvas.pack()

        # Add cells to the canvas 
        square_size = self.width / length
        print(square_size)
        # For each row in the CA data 
        for row in range(len(ca_time_space)):
            lattice = ca_time_space[row]
            # For each cell in that row 
            for col in range(length):
                color = 'white' if (lattice[col] == '0') else 'black'
                x0 = square_size * col
                y0 = square_size * row
                x1 = x0 + square_size 
                y1 = y0 + square_size 
                canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline = color)
    
