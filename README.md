# Dự Án Xác Thực Khuôn Mặt (Face Authentication)

## Mô Tả
Dự án xác thực khuôn mặt sử dụng các thuật toán nhận dạng khuôn mặt để xác thực người dùng. Dự án này giúp nhận diện và xác thực người dùng thông qua việc sử dụng AIAI so sánh khuôn mặt từ camera với dữ liệu mẫu đã được lưu trữ trước đó.

## Các Bước Cài Đặt

### Bước 1: Cài Đặt Python

Đảm bảo bạn đã cài đặt Python 3.9 - 3.12 (Recommended: 3.11.1). Bạn có thể tải Python tại đây: [Python Downloads](https://www.python.org/downloads/).

Để kiểm tra xem Python đã được cài đặt chưa, sử dụng lệnh:

```bash
python --version
```
### Bước 2: Tạo và kích hoạt môi trường ảo (Tùy chọn)
```bash
python -m venv venv
```
#### Kích hoạt môi trường ảo Windows:
```bash
venv\Scripts\activate
```
#### Nếu dùng macOS / Linux:
```bash
source venv/bin/activate
```
### Bước 3: Cài đặt thư viện từ File requirements.txt
```bash
pip install -r requirements.txt
```
#### Để chạy chế độ hình ảnh và video với giao diện Gradio:
```bash
python face_authentication_gradio.py
```
#### Để chạy chế độ realtime với chế độ webcam realtime:
```bash
python face_authentication_Flask.py
```


## Cài đặt và chạy real time với Pi4: 
### Bước 1: Tạo và kích hoạt môi trường ảo (Tùy chọn)
```bash
python -m venv venv
```
#### Kích hoạt môi trường ảo:
```bash
source venv/bin/activate
```
### Bước 2: Cài đặt thư viện ttừ File requirements_Pi.txt
```bash
pip install -r requirements_Pi.txt
```
### Bước 3: Chạy chế độ Realtime với chế độ Webcam:
```bash
python face_authentication_Pi.py
```
#### Các chế độ bao gồm: 
r: Đăng ký khuôn mặt mới 
v: Xác thực
q: Thoát 
