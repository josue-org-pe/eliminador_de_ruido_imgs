import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2


class DesruidadorSimple:
    def __init__(self, root):
        self.root = root
        self.root.title("PRIMER PROTOTIPO ")
        self.root.geometry("900x600")
        
        self.imagen_original = None
        self.imagen_ruidosa = None
        self.imagen_limpia = None
        
        self.photo_orig = None
        self.photo_limpia = None
        
        self.crear_interfaz()
    #esto es lo del framework , tkinder para ui local noma , creacion de botonoes y ubicaciones 
    def crear_interfaz(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Botones
        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="SUBIR  IMAGEN", command=self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="SIMULAR RUIDO", command=self.agregar_ruido).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="ELIMINAR RUIDO", command=self.eliminar_ruido).pack(side=tk.LEFT, padx=5)
        
        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas_orig = tk.Canvas(canvas_frame, width=400, height=400, bg='lightgray')
        self.canvas_orig.pack(side=tk.LEFT, padx=10)
        
        self.canvas_limpia = tk.Canvas(canvas_frame, width=400, height=400, bg='lightgray')
        self.canvas_limpia.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(canvas_frame, text="Original").place(in_=self.canvas_orig, relx=0.5, rely=1.02, anchor="n")
        ttk.Label(canvas_frame, text="Sin Ruido").place(in_=self.canvas_limpia, relx=0.5, rely=1.02, anchor="n")
    #funcion para cargar imgangenes 
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not ruta:
            return
        
        try:
            self.imagen_original = Image.open(ruta).convert('L')  # escala de grises
            self.imagen_ruidosa = None
            self.imagen_limpia = None
            self.mostrar_imagen(self.imagen_original, self.canvas_orig, "orig")
            self.mostrar_imagen(None, self.canvas_limpia, "limpia")
            messagebox.showinfo("Éxito", "Imagen cargada. Puedes agregar ruido o procesar directamente si ya está ruidosa.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")
    
    def agregar_ruido(self):
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        
        img_array = np.array(self.imagen_original)
        ruido = np.random.normal(0, 25, img_array.shape)  # sigma=25 → ruido moderado
        ruidosa = np.clip(img_array + ruido, 0, 255).astype(np.uint8)
        self.imagen_ruidosa = Image.fromarray(ruidosa, mode='L')
        self.mostrar_imagen(self.imagen_ruidosa, self.canvas_orig, "orig")
    
    def eliminar_ruido(self):
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        
        img_to_denoise = self.imagen_ruidosa if self.imagen_ruidosa else self.imagen_original
        img_array = np.array(img_to_denoise)
        
        try:
            limpia_array = cv2.fastNlMeansDenoising(img_array, None, h=10, templateWindowSize=7, searchWindowSize=21)
        except Exception:
            limpia_array = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
        
        self.imagen_limpia = Image.fromarray(limpia_array, mode='L')
        self.mostrar_imagen(self.imagen_limpia, self.canvas_limpia, "limpia")
        messagebox.showinfo("Éxito", "¡Ruido eliminado con éxito!")
    
    def mostrar_imagen(self, imagen, canvas, tipo):
        canvas.delete("all")
        if imagen is None:
            canvas.create_text(200, 200, text="Sin imagen", fill="gray")
            if tipo == "orig":
                self.photo_orig = None
            else:
                self.photo_limpia = None
            return
        
        img_display = imagen.resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_display)
        
        if tipo == "orig":
            self.photo_orig = photo
        else:
            self.photo_limpia = photo
        
        canvas.create_image(200, 200, image=photo)


def main():
    root = tk.Tk()
    app = DesruidadorSimple(root)
    root.mainloop()


if __name__ == "__main__":
    main()