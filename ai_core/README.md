# AI Core - Jetson Deployment Guide

Tài liệu này hướng dẫn cách sử dụng Docker trên phần cứng NVIDIA Jetson Nano nhằm hỗ trợ khả năng tăng tốc phần cứng GPU một cách tối ưu nhất.

## 🐳 Hướng dẫn chạy và thao tác với Docker

### Bước 1: Build Docker Image (Chạy lần đầu)
Mở terminal trên Jetson Nano, đi đến folder `ai_core` và chạy lệnh sau để đóng gói môi trường (Quá trình này có thể tốn một khoảng thời gian tuỳ thuộc vào tốc độ mạng):
```bash
docker compose build
```

### Bước 2: Khởi động hệ thống Docker (Chế độ chạy nền)
Lệnh này sẽ kích hoạt Docker và giữ cho container luân phiên hoạt động ngầm (detached mode) nhờ lệnh `tail -f /dev/null`:
```bash
docker compose up -d
```

### Bước 3: Truy cập vào Terminal (Bash) bên trong Docker
Đây là bước quan trọng nhất theo như thiết kế sửa đổi từ trước. Thay vì tự chạy code lúc khởi động, bạn có thể tự do "bước vào" môi trường riêng của Docker bằng lệnh:
```bash
docker exec -it dat_lab_aicore /bin/bash
```

> **Mẹo:**
> - Ngay sau khi vào, bạn sẽ đứng ở thư mục `/workspace/ai_core` (nơi chứa toàn bộ mã nguồn).
> - Từ đây, bạn có thể chủ động gõ `python3 main.py` để test, dùng lệnh `pip install ...` hoặc chỉnh sửa file. Do thư mục này được ánh xạ (mount) trực tiếp với máy tính gốc, code chạy sẽ ăn ngay lập tức.
> - Nếu muốn thoát môi trường Docker về lại Jetson Nano, hãy gõ lệnh `exit`.

---

## Các lệnh hỗ trợ khác (Quản lý tiến trình)
- **Xem dòng chữ in ra màn hình (Logs):** Hữu ích nếu sau này bạn đổi ý chạy thẳng code python lúc khởi động `CMD` mà không muốn chui vào Terminal:
  ```bash
  docker logs -f dat_lab_aicore
  ```
  *(Nhấn `Ctrl + C` để thoát màn hình xem log)*

- **Dập, tắt và xóa container (Khi không dùng nữa):**
  ```bash
  docker compose down
  ```
