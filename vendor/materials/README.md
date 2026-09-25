# Tài sản cho trình đọc tài liệu

Các tệp trong danh mục này và các thư mục liên quan được phục vụ cục bộ. Không dùng CDN khi chạy.

## Marked 18.0.7

- Dự án: `https://github.com/markedjs/marked`
- Gói tải: `https://registry.npmjs.org/marked/-/marked-18.0.7.tgz`
- Ngày tải: `2026-08-30`
- Giấy phép: MIT; bản sao tại `../marked/18.0.7/LICENSE`
- Tệp chạy: `../marked/18.0.7/marked.umd.js`
- SHA-256 gói: `edcb08b6a56e5bfa9a1177a29943f19587771297e123694134d166546e1201b6`
- SHA-256 tệp chạy: `7a1f8c5e7226b75ff16644bdb2c0130d2ae7371e7ea3106c2d6dac77ab0ff7b6`
- Vai trò: chuyển Markdown thành HTML trong trình duyệt.

## DOMPurify 3.4.7

- Dự án: `https://github.com/cure53/DOMPurify`
- Gói tải: `https://registry.npmjs.org/dompurify/-/dompurify-3.4.7.tgz`
- Ngày tải: `2026-08-30`
- Giấy phép: MPL-2.0 hoặc Apache-2.0; bản sao Apache-2.0 tại `../dompurify/3.4.7/LICENSE`
- Tệp chạy: `../dompurify/3.4.7/purify.min.js`
- SHA-256 gói: `a5096949288f85f5201eee7908adb755305e9ddb29bd66a4e9d703f12126d22f`
- SHA-256 tệp chạy: `f84e522876a6cfadecb89c173356409acec39f580c69018559c9a50e96299b0c`
- Vai trò: làm sạch HTML trước khi gắn vào DOM.
- Cấu hình dự án không dùng `IN_PLACE`, hook toàn cục hoặc `setConfig()`.

## KaTeX 0.16.22

- Tài sản đã có trong `../katex/` và được dùng chung với RevealJS.
- Phiên bản được xác minh từ `../katex/dist/katex.js`.
- Giấy phép: MIT; bản sao tại `../katex/LICENSE`.
- SHA-256 `katex.min.js`: `e8d885505949f3a5f4abdd5dd0d53696bd1371ad26ffbf4f310dcd77c8cdae89`
- SHA-256 `auto-render.min.js`: `bb53eb953394531aae36fdd537065c4244eb8542901a3ce914601d932675b8ac`
- SHA-256 `katex.min.css`: `19095127357ed6d29fe0a63a6b000c913a89f7f1963b765dd3715e97c9852e75`
- Vai trò: render `$...$` và `$$...$$` sau khi HTML đã được làm sạch.
- KaTeX tạo một số thuộc tính kiểu nội tuyến khi render; vì vậy CSP của viewer cho phép `style-src 'unsafe-inline'` nhưng vẫn giữ `script-src 'self'` và làm sạch HTML trước khi gọi KaTeX.

Trình nạp `material-bootstrap.js` dùng SRI SHA-256 tương ứng với năm checksum ở trên khi chạy qua HTTP/HTTPS. Khi mở trực tiếp bằng `file://`, trình duyệt không hỗ trợ kiểm tra SRI cho tệp local, nên trình nạp bỏ thuộc tính này và tải đúng các tài sản cục bộ đã được kiểm kê. Khi cập nhật vendor, phải cập nhật đồng thời checksum trong danh mục và trình nạp.
