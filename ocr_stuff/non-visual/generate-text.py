import easyocr
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
language = 'lt'
img = 'test.png'

reader = easyocr.Reader([language])
result = reader.readtext(img, detail=0)
print(result)
with open("text.txt", "w", encoding="utf-8") as f:
	f.write(" ".join(result))
