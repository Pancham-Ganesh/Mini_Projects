import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import socket

def voteCast(root, frame1, vote, client_socket):
    for widget in frame1.winfo_children():
        widget.destroy()

    client_socket.send(vote.encode())  # Send vote

    message = client_socket.recv(1024).decode()  # Receive response
    print(message)

    if message == "Successful":
        Label(frame1, text="Vote Casted Successfully", font=('Helvetica', 18, 'bold')).grid(row=1, column=1)
    else:
        Label(frame1, text="Vote Cast Failed...\nTry again", font=('Helvetica', 18, 'bold')).grid(row=1, column=1)

    client_socket.close()


def votingPg(root, frame1, client_socket):
    root.title("Cast Vote")
    for widget in frame1.winfo_children():
        widget.destroy()

    Label(frame1, text="Cast Vote", font=('Helvetica', 18, 'bold')).grid(row=0, column=1, rowspan=1)
    Label(frame1, text="").grid(row=1, column=0)

    vote = StringVar(frame1, "-1")

    # Load and keep image references
    bjpLogo = ImageTk.PhotoImage(Image.open("img/bjp.png").resize((45, 45), Image.Resampling.LANCZOS))
    congLogo = ImageTk.PhotoImage(Image.open("img/cong.jpg").resize((35, 48), Image.Resampling.LANCZOS))
    aapLogo = ImageTk.PhotoImage(Image.open("img/aap.png").resize((55, 40), Image.Resampling.LANCZOS))
    ssLogo = ImageTk.PhotoImage(Image.open("img/ss.png").resize((50, 45), Image.Resampling.LANCZOS))
    notaLogo = ImageTk.PhotoImage(Image.open("img/nota.jpg").resize((45, 35), Image.Resampling.LANCZOS))

    # BJP
    bjpImg = Label(frame1, image=bjpLogo)
    bjpImg.image = bjpLogo
    bjpImg.grid(row=2, column=0)
    Radiobutton(frame1, text="BJP\n\nNarendra Modi", variable=vote, value="bjp", indicator=0, height=4, width=15,
                command=lambda: voteCast(root, frame1, "bjp", client_socket)).grid(row=2, column=1)

    # Congress
    congImg = Label(frame1, image=congLogo)
    congImg.image = congLogo
    congImg.grid(row=3, column=0)
    Radiobutton(frame1, text="Congress\n\nRahul Gandhi", variable=vote, value="cong", indicator=0, height=4, width=15,
                command=lambda: voteCast(root, frame1, "cong", client_socket)).grid(row=3, column=1)

    # AAP
    aapImg = Label(frame1, image=aapLogo)
    aapImg.image = aapLogo
    aapImg.grid(row=4, column=0)
    Radiobutton(frame1, text="Aam Aadmi Party\n\nArvind Kejriwal", variable=vote, value="aap", indicator=0, height=4, width=15,
                command=lambda: voteCast(root, frame1, "aap", client_socket)).grid(row=4, column=1)

    # Shiv Sena
    ssImg = Label(frame1, image=ssLogo)
    ssImg.image = ssLogo
    ssImg.grid(row=5, column=0)
    Radiobutton(frame1, text="Shiv Sena\n\nUdhav Thakrey", variable=vote, value="ss", indicator=0, height=4, width=15,
                command=lambda: voteCast(root, frame1, "ss", client_socket)).grid(row=5, column=1)

    # NOTA
    notaImg = Label(frame1, image=notaLogo)
    notaImg.image = notaLogo
    notaImg.grid(row=6, column=0)
    Radiobutton(frame1, text="\nNOTA    \n  ", variable=vote, value="nota", indicator=0, height=4, width=15,
                command=lambda: voteCast(root, frame1, "nota", client_socket)).grid(row=6, column=1)

    frame1.pack()
    root.mainloop()


# Example test usage:
# if __name__ == "__main__":
#     root = Tk()
#     root.geometry('500x600')
#     frame1 = Frame(root)
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect(('localhost', 9999))  # Adjust IP/port if needed
#     votingPg(root, frame1, client_socket)
