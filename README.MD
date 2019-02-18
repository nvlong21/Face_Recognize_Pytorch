# Face recognition
## Requirements
Python 3.5+

Commanline:
```
pip3 install -r requirements.txt
```

## Usage:
### Download
```
git clone https://github.com/vanlong96tg/face_recognize
mkdir face_recognize/weights
cd face_recognize/weights
wget https://www.dropbox.com/s/akktsgxp0n8cwn2/model_mobilefacenet.pth?dl=0 -O model_mobilefacenet.pth
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth
```
### Python:
Run with default threshold=1.34:
```
python3 face_verify.py -path {path_folder_image} -image {path_image}
```
Run with threshold: 
```
python3 face_verify.py -path {path_folder_image} -image {path_image} -threshold {threshold}
```
Use model ir_se50 (slower but more accurate): 
```
python3 face_verify.py -path {path_folder_image} -image {path_image} -ismobi 0
```
Use model mtcnn for face detection (faster but more errors): 
```
python3 face_verify.py -path {path_folder_image} -image {path_image} -user_mtcnn 1
```
Run with full option:
```
python3 face_verify.py -path {path_folder_image} -image {path_image} -threshold 1.4 -ismobi 0 -user_mtcnn 0
```
### Docker:
Install docker
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER
```
Install docker-compose
```
pip3 install docker-compose
```
Run
```
docker-compose up --build -d
```
## Test API with docker
Requirements: url not authenticate

Test with Postman
### Two image detection
**URL:** http://localhost:8084/face_recognition
```
{
	"image_url_origin":"https://www.dropbox.com/s/vm8fvi9xdmjrdmr/PQH_0000.png?dl=0",
	"image_url_detection":"https://www.dropbox.com/s/vm8fvi9xdmjrdmr/PQH_0000.png?dl=0"
}
```
### Multiple image 
Please zip image to file

**URL:** http://localhost:8084/face_recognition_two_image
```
{
	"image_url":"https://www.dropbox.com/s/vm8fvi9xdmjrdmr/PQH_0000.png?dl=0",
	"file_zip_url":"https://www.dropbox.com/s/bf705wgk2n9vog6/test.zip?dl=0"
}
```
### Training:
* Performance

	|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
	|:---:|:---:|:---:|:---:|
	|99.61|99.43|97.32|94.88|

### Acknowledgement 
* This repo is inspired by [InsightFace.MXNet](https://github.com/deepinsight/insightface), [InsightFace.PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch), [ArcFace.PyTorch](https://github.com/ronghuaiyang/arcface-pytorch), [MTCNN.MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection) and [PretrainedModels.PyTorch](https://github.com/Cadene/pretrained-models.pytorch).
* Web Face Recognition Training Datasets [InsightFace.MXNet](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)