import pillowfight
import PIL

PATH = r"test.jpg"
in_img = PIL.Image.open(PATH)

out_img = pillowfight.ace(in_img)
out_img.show()