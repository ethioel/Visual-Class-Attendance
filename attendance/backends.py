class DlibBackend:                
    def encode(self, rgb, boxes):
        import face_recognition
        return face_recognition.face_encodings(rgb, boxes)

class FacenetPytorchBackend:             # 512-d embeddings, VGGFace2-trained
    def __init__(self):
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    def encode(self, rgb, boxes):
        import torch
        from facenet_pytorch import fixed_image_standardization
        faces = [fixed_image_standardization(self.crop_aligned(rgb, b)) for b in boxes]
        with torch.no_grad():
            return self.model(torch.stack(faces)).numpy()