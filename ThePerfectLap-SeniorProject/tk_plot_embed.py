import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Tkinter + Matplotlib Smoke Test")
root.geometry("600x400")

fig = Figure(figsize=(5,3))
ax = fig.add_subplot(111)
ax.plot([0,1,2,3],[0,1,4,9], marker="o")
ax.set_title("Embedded Plot")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
