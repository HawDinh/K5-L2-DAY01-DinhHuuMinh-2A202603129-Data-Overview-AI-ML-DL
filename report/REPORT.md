# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python `3.13.18` / PyTorch `2.11.0+cpu` / Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`. Riêng phần so sánh ngưỡng của detection chạy thêm `0.20` và `0.60`, không thay đổi ngưỡng tạo JSON.

**Thay đổi so với notebook nguồn:** Không thay đổi checkpoint, taxonomy, threshold hay mã dự đoán. Có chạy thêm cell để đóng gói và tải bản sao `day1_lab_outputs/` về máy phục vụ viết báo cáo; không thay đổi nội dung các file bằng chứng.

> **Nguồn bằng chứng:** Các file JSON, PNG và `IMAGE_ATTRIBUTION.md` trong `day1_lab_outputs/` do notebook xuất ra. Ba ảnh COCO có `sample_id`/`coco_image_id`: `traffic`/`210273`, `kitchen`/`397133`, `dining`/`166918`; checksum ảnh đã được notebook kiểm tra `PASS` ở cell tải dữ liệu. Đây là **prediction** từ checkpoint đã huấn luyện sẵn, không phải ground truth.

### Từ cần nhớ

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả dự đoán của mô hình, chưa được xác minh là nhãn đúng |
| Ground truth | Nhãn do annotator tạo và reviewer kiểm tra theo guideline |
| Taxonomy | Danh sách lớp quy định trước (ví dụ ImageNet-1K, COCO-80) |
| Checkpoint | Trọng số mô hình đã huấn luyện, lưu trong tệp `.pt` |
| Guideline | Quy tắc gán nhãn thống nhất |
| Annotator | Người thực hiện gán nhãn |
| Reviewer / QC | Người kiểm tra tính đúng và tính nhất quán của nhãn |
| Escalation | Chuyển trường hợp chưa rõ quy tắc cho Lab Coach xử lý |
| Model score | Điểm số mô hình xuất ra, không phải điểm chất lượng ground truth |

## 1. Phân loại ảnh

1. **Record hạng 1 của ảnh `traffic`:** `class_id=468`; `class_name=cab`; `rank=1`; `score=0.510915`; `taxonomy_name=ImageNet-1K` (nguồn: `classification_predictions.json`, `sample_id=traffic`, `rank=1`).

2. Record này mô tả **toàn bộ bức ảnh** vì tác vụ image classification gắn một nhãn dự đoán cho ảnh đầu vào, không sinh hộp tọa độ hay một instance riêng. Nhãn `cab` là lớp xếp hạng cao nhất của cả ảnh `traffic`, không có nghĩa mọi xe trong ảnh đều là taxi.

3. Danh sách tên lớp đến từ **taxonomy đi kèm checkpoint** (ở đây là ImageNet-1K), không phải do mô hình tự nghĩ ra khi dự đoán.

4. `class_id` là mã lớp để hệ thống đối chiếu; `class_name` giúp người đọc hiểu nhãn; `taxonomy_name` xác định bộ lớp mà mã và tên đang thuộc về. Chỉ có tên lớp có thể gây nhầm lẫn nếu các taxonomy khác nhau dùng tên giống hoặc gần giống nhau, hoặc đánh số lớp khác nhau.

5. Với ảnh có nhiều vật thể và chủ đề, guideline cần xác định **tiêu chí chọn đúng một nhãn**: ưu tiên đối tượng/chủ đề chính theo mục tiêu dữ liệu, cách xử lý nhiều đối tượng cùng nổi bật, ảnh không có lớp phù hợp, và khi nào phải đánh dấu ambiguous/escalate. Không tự chọn theo score của mô hình.

6. `score=0.510915` chỉ thể hiện mức điểm mô hình gán cho `cab`; nó không chứng minh lớp đó đúng và không đo chất lượng nhãn người. Ground truth cần được annotator đối chiếu ảnh và guideline, sau đó QC theo quy trình.

## 2. Phát hiện vật thể

1. **Một record thuộc ảnh `kitchen`:** `class_name=person`; `class_id=0`; `score=0.912625`; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`; `bbox_width=113.58`; `bbox_height=279.68` (đơn vị pixel; nguồn: `detection_predictions.json`, `sample_id=kitchen`, record `person` có score `0.912625`).

2. Box dùng hệ tọa độ gốc `(0, 0)` ở góc trên trái ảnh: góc **trên trái** tại `(385.33, 69.24)`, góc **dưới phải** tại `(498.92, 348.92)`. Box rộng xấp xỉ **113.58 pixel** và cao **279.68 pixel** (số trong JSON có làm tròn riêng nên hiệu tọa độ hiển thị có thể chênh 0.01 pixel). Đây là vùng chữ nhật mô hình dự đoán bao quanh người trong bếp; không thể hiện đường viền chi tiết.

3. Với riêng ảnh `kitchen`, ở ngưỡng `0.20` có **17 prediction**, còn ngưỡng `0.60` có **6 prediction**. Ngưỡng tạo file JSON mặc định `0.35` có **11 prediction** (nguồn: output cell so sánh threshold của notebook).

4. Hạ ngưỡng làm tăng số box phải xem. Reviewer cần đối chiếu từng box với ảnh, loại/đánh dấu false positive hoặc box trùng, kiểm tra lớp và vị trí hộp, đồng thời phát hiện vật thể bị bỏ sót. **Nhiều prediction hơn không đồng nghĩa nhiều nhãn đúng hơn.**

5. **Quy tắc vẽ box đề xuất:** Mỗi vật thể hợp lệ phải có **một box chữ nhật khít với phần nhìn thấy của vật thể**, theo tọa độ pixel `xyxy`; không gom nhiều đối tượng riêng thành một box, không vượt biên ảnh và không tự mở rộng box để đoán phần bị che nếu guideline chỉ yêu cầu phần nhìn thấy. Khi gặp trường hợp che khuất, cần dùng cùng một chính sách nhất quán cho cả tập dữ liệu.

6. Ở mép trái ảnh `kitchen`, mô hình dự đoán `person` với `score=0.610931` và `bbox_xyxy=[0.12,263.18,61.55,310.78]`, nhưng hình chỉ cho thấy một phần bàn tay/cánh tay; không thể xem đây là ground truth về một người đầy đủ (nguồn: `detection_predictions.json`, `detection_predictions.png`). Tôi đề xuất **guideline chung**: chỉ gán box cho vật thể bị cắt mép khi phần nhìn thấy đủ để nhận diện theo tiêu chí đã quy định; box giới hạn trong ảnh và không suy đoán phần ngoài khung. Nếu guideline chưa xác định mức độ hiển thị tối thiểu hoặc cách xử lý trường hợp chỉ thấy một bộ phận cơ thể, phải **escalate cho Lab Coach** trước khi quyết định gán nhãn.

## 3. Phân đoạn theo từng đối tượng

1. **Một record thuộc ảnh `kitchen`:** `instance_id=kitchen-001`; `class_id=0`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; `bbox_xyxy=[385.45, 66.44, 498.02, 348.58]`. Năm điểm đầu của `polygon_xy` theo đúng thứ tự trong JSON: `[[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]]` (đơn vị pixel; nguồn: `segmentation_predictions.json`, `instance_id=kitchen-001`).

2. Box chỉ cho vùng chữ nhật bao ngoài, còn **polygon mô tả đường viền và diện tích của chính đối tượng**: thể hiện hình dạng bất quy tắc và phân biệt pixel thuộc đối tượng với nền trong box. Vì thế, polygon hữu ích khi nhiều vật thể chạm nhau hoặc có khoảng trống trong box.

3. `instance_id` phân biệt **từng đối tượng cụ thể trong bộ output**: chẳng hạn hai người có thể cùng `class_id=0` nhưng phải có instance riêng. Nó không phải **class ID** (mã loại vật thể) và cũng không phải **tracking ID** (định danh dùng để theo dõi đối tượng xuyên nhiều khung hình video).

4. **Quy tắc vẽ biên mask đề xuất:** Vẽ polygon theo ranh giới **phần vật thể thực sự nhìn thấy** ở độ phân giải ảnh, không lấy nền hoặc vật thể khác vào mask, không tự nối qua vùng bị che để đoán phần khuất; mỗi đối tượng có mask/instance riêng. Các trường hợp biên khó phân biệt phải theo cùng quy tắc QC hoặc chuyển Lab Coach nếu guideline chưa có quyết định.

5. Trong ảnh `kitchen`, đường viền **người đứng bên phải** tiếp giáp nhiều dụng cụ nấu ăn/đồ vật treo ở nền; ranh giới quanh tóc, vai và tay có thể khó tách nền (xem `segmentation_prediction.png`, instance `kitchen-001`). Tôi dùng **guideline về visible-mask** để bám phần người nhìn thấy và loại vật dụng khỏi mask; nếu ảnh quá mờ khiến không thể xác định đường biên nhất quán hoặc quy tắc xử lý che khuất không rõ, sẽ **escalate cho Lab Coach**. Không coi đường biên do model tô là ground truth sẵn có.

## 4. Vòng đời và kiểm tra chất lượng

Bài lab này chỉ quan sát prediction. Trong hệ thống sản xuất, có thể liên hệ: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **Một class cho toàn ảnh** theo taxonomy và quy tắc chọn chủ đề chính. | Ảnh `traffic` có nhiều xe, nhưng top-1 dự đoán `cab` (`score=0.510915`); cần thống nhất ảnh nhiều đối tượng nên chọn lớp nào. Đây là **điểm mơ hồ**, chưa đủ căn cứ kết luận dự đoán sai. | Xem toàn ảnh, chọn một lớp theo mục tiêu/guideline; đánh dấu ca không rõ. | Đối chiếu lớp với nội dung ảnh, taxonomy và quy tắc ảnh nhiều chủ đề; không dựa riêng model score. |
| Phát hiện vật thể | **Một cặp class + bounding box cho từng đối tượng** cần gán nhãn. | Ảnh `kitchen` có prediction `person` (`score=0.610931`) tại mép trái nhưng chỉ thấy một phần tay; cần quyết định liệu mức độ hiển thị này đủ để gán nhãn hay không. Số prediction thay đổi `17 → 6` khi threshold tăng `0.20 → 0.60`. | Xác định từng vật thể hợp lệ, vẽ box khít theo chính sách cắt mép/che khuất. | Kiểm tra thiếu/thừa box, box trùng, nhầm class, kích thước và vị trí box, nhất là vùng cắt mép. |
| Instance segmentation | **Một class và mask/polygon riêng cho mỗi instance** (thường kèm box). | Biên `person` (`kitchen-001`) tiếp giáp các dụng cụ/nền bếp; model tạo polygon **348 điểm**, nhưng nhiều điểm không tự bảo đảm đường biên chính xác. | Vẽ mask theo biên phần nhìn thấy, tách riêng các instance; đánh dấu biên không rõ. | Kiểm tra mask ăn vào nền, thiếu vùng người, gộp nhầm đối tượng, biên che khuất và tính thống nhất với guideline. |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu:** Chỉ sử dụng ba ảnh COCO công khai được notebook cung cấp và các output được phép; không đưa ảnh cá nhân, dữ liệu khách hàng/nội bộ hoặc dữ liệu nhạy cảm lên Colab hay repository GitHub công khai.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi được cho phép, tôi sẽ **dừng sử dụng/đăng tải**, không tự xử lý để hợp thức hóa dữ liệu và **báo Lab Coach (hoặc GV/Lab Lead)** để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [ ] `REPORT.md` đã điền và nằm trong thư mục `report/` trên GitHub
- [ ] `day1_lab_outputs/classification_predictions.json` đã upload vào `report/`
- [ ] `day1_lab_outputs/detection_predictions.json` đã upload vào `report/`
- [ ] `day1_lab_outputs/segmentation_predictions.json` đã upload vào `report/`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md` đã upload vào `report/`
- [ ] `day1_lab_outputs/visuals/classification_top5.png` đã upload vào `report/`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png` đã upload vào `report/`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png` đã upload vào `report/`
- [x] Ô validation evidence trong notebook báo `PASS: đủ 3 JSON, 3 PNG, attribution và instance_id duy nhất.`
- [x] Nội dung báo cáo không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm
- [ ] Đã nộp **link repository** trên LMS

> Các mục GitHub/LMS chỉ đánh dấu hoàn thành sau khi thực sự upload và nộp. `REPORT.md` không phải file ground truth và không thay thế bước review của Lab Coach.
