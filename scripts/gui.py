import tkinter as Tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import librosa.display
import display
import gui_util
import stft_zoom

from tkinter.filedialog import askopenfilename

def openfile(axis, figure, sr=44100):

   path = askopenfilename(parent=root)
   y = gui_util.load_audio(path)
   y = y[:30*sr]
   draw_spec(y, axis, figure)

def draw_spec(y, axis, figure, sr=44100):
	D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512)), ref=np.max)
	x_data, y_data = gui_util.get_axes_values(sr, 0, [0, len(y)/sr], D.shape)
	display.specshow(D, x_data, y_data, ax=axis, y_axis='log')
	canvas = FigureCanvasTkAgg(figure, master=root)
	canvas.show()
	canvas.get_tk_widget().grid(row=0, columnspan=5)




# D, axis = gui_util.get_spectrogram(y)
# display.specshow(D, axis[0], axis[1], ax=a, y_axis='log')

def process_zoom():

	if not E1.get() or not E2.get() or not E3.get() or not E4.get():
		return print('deu ruim')

	freq_range = [0, 0]
	freq_range[0] = float(E1.get())
	freq_range[1] = float(E2.get())

	time_range = [0, 0]
	time_range[0] = float(E3.get())
	time_range[1] = float(E4.get())
	
	return(draw_spec_zoom(stft_zoom.stft_zoom(y, freq_range, time_range, 44100)))


# D_zoom = 0
# new_sr = 0
# f_min  = 0

def draw_spec_zoom(zoom):
	D = zoom[0]
	x = zoom[1]
	y = zoom[2]

	zoom_window = Tk.Toplevel(root)
	zoom_window.wm_title("Zoom Detail")

	f = Figure()
	a = f.add_subplot(111)

	display.specshow(D, x, y, ax=a)

	canvas = FigureCanvasTkAgg(f, master=zoom_window)
	canvas.show()
	canvas.get_tk_widget().grid(row=0, columnspan=5)

# def save_zoom(zoom):
# 	# global D_zoom, new_sr, f_min
# 	# D_zoom = zoom[0]
# 	# new_sr = zoom[1]
# 	# f_min = zoom[2]

# 	# x_data, y_data = gui_util.get_axes_values(new_sr, f_min, time_range, D_zoom.shape)
# 	return(draw_spec_zoom(zoom[0], zoom[1], zoom[2]))

	# print(zoom)

root = Tk.Tk()
root.wm_title("STFT Zoom Tool")

L1 = Tk.Label(master=root, text="Frequency range: ")
L1.grid(row=1, column=0)
E1 = Tk.Entry(master=root, bd = 5)
E1.grid(row=1, column=1)

L2 = Tk.Label(master=root, text=" to ")
L2.grid(row=1, column=2)
E2 = Tk.Entry(master=root, bd = 5)
E2.grid(row=1, column=3)

L3 = Tk.Label(master=root, text="Time range: ")
L3.grid(row=2, column=0)
E3 = Tk.Entry(master=root, bd = 5)
E3.grid(row=2, column=1)

L4 = Tk.Label(master=root, text=" to ")
L4.grid(row=2, column=2)
E4 = Tk.Entry(master=root, bd = 5)
E4.grid(row=2, column=3)

L5 = Tk.Label(master=root, text="Hz")
L5.grid(row=1, column=4)
L6 = Tk.Label(master=root, text="seconds")
L6.grid(row=2, column=4)

B = Tk.Button(master=root, text="Zoom in...", command=process_zoom)
B.grid(row=3, columnspan=5)

f = Figure()
a = f.add_subplot(111)

# y = gui_util.get_audio('data/gymno.wav')
# D, axis = gui_util.get_spectrogram(y)
# display.specshow(D, axis[0], axis[1], ax=a, y_axis='log')
# librosa.display.specshow(D, sr=44100, hop_length=512/4 , ax=a, y_axis='log', x_axis='time')

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().grid(row=0, columnspan=5)

menubar = Tk.Menu(root)
filemenu = Tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command= lambda: openfile(a, f))
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)

Tk.mainloop()