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
	global y
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

def process_zoom():

	if not E1.get() or not E2.get() or not E3.get() or not E4.get():
		return print('error: unspecified parameters')

	freq_range = [0, 0]
	freq_range[0] = float(E1.get())
	freq_range[1] = float(E2.get())

	time_range = [0, 0]
	time_range[0] = float(E3.get())
	time_range[1] = float(E4.get())
	
	return(draw_spec_zoom(stft_zoom.stft_zoom(y, freq_range, time_range, 44100)))

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

root = Tk.Tk()
root.wm_title("STFT Zoom Tool")

# Top menu for opening files
menubar = Tk.Menu(root)
filemenu = Tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command= lambda: openfile(a, f))
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

# Frame for plotting main spectrogram
plotframe = Tk.Frame(root)
plotframe.grid(row=0, columnspan=10)
f = Figure()
a = f.add_subplot(111)
canvas = FigureCanvasTkAgg(f, master=plotframe)
canvas.show()
canvas.get_tk_widget().grid(row=0, columnspan=10)

# Frame for specifying frequency range and time range
rangeframe = Tk.Frame(root, bd=20)
rangeframe.grid(column=0,row=1)

L1 = Tk.Label(master=rangeframe, text="Frequency range", bd=2)
L1.grid(row=0, column=0, columnspan=4)
E1 = Tk.Entry(master=rangeframe, bd = 5, width=8)
E1.grid(row=1, column=1)

L2 = Tk.Label(master=rangeframe, text=" to ")
L2.grid(row=1, column=2)
E2 = Tk.Entry(master=rangeframe, bd = 5, width=8)
E2.grid(row=1, column=3)

L3 = Tk.Label(master=rangeframe, text="Time range", bd=2)
L3.grid(row=2, column=0, columnspan=4)
E3 = Tk.Entry(master=rangeframe, bd = 5, width=8)
E3.grid(row=3, column=1)

L4 = Tk.Label(master=rangeframe, text=" to ")
L4.grid(row=3, column=2)
E4 = Tk.Entry(master=rangeframe, bd = 5, width=8)
E4.grid(row=3, column=3)

L5 = Tk.Label(master=rangeframe, text="Hz")
L5.grid(row=1, column=4)
L6 = Tk.Label(master=rangeframe, text="seconds")
L6.grid(row=3, column=4)

# Frame for specifying resolution
resframe = Tk.Frame(root, bd=20)
resframe.grid(column=1, row=1)

L6 = Tk.Label(master=resframe, text="Resolution")
L6.grid(row=0)

E5 = Tk.Entry(master=resframe, bd = 5, width=8)
E5.grid(row=1, column=0)

E6 = Tk.Entry(master=resframe, bd = 5, width=8)
E6.grid(row=2, column=0)

timeop = Tk.StringVar(root)
freqop = Tk.StringVar(root)
choices_time = { 'time frames','ms per bin'}
choices_freq = { 'freq. bins','Hz per bin'}
timeop.set('time frames')
freqop.set('freq. bins')

popupMenu1 = Tk.OptionMenu(resframe, timeop, *choices_time)
popupMenu1.grid(row=2, column=1)
popupMenu2 = Tk.OptionMenu(resframe, freqop, *choices_freq)
popupMenu2.grid(row=1, column=1)

B = Tk.Button(master=root, text="Zoom in...", command=process_zoom)
B.grid(row=2, columnspan=5)

Tk.mainloop()