import cv2
from torchvision import transforms

def read_categories(file_path):
    categories = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.strip().split('.', 1)
            if len(parts) == 2:
                key, value = parts
                categories[str(index)] = value.strip()
    return categories

def url_to_tensor(image_url):
    img = cv2.imread(image_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
    ])
    img = transform(img)
    return img
