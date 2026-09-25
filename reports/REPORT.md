# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Văn A

Công cụ gán nhãn đã dùng: CVAT

Sao chép file này thành `reports/REPORT.md` rồi điền vào các chỗ ĐIỀN. Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Camera đứng một chỗ, một chiếc xe nằm trong hình vài giây. Ảnh học và ảnh kiểm tra phải cách nhau theo thời gian. Nếu trộn ngẫu nhiên, cùng một xe có thể vừa được AI học vừa được dùng để chấm. Điểm sẽ đẹp hơn sự thật.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Số chính là điểm khớp khung 0.771. Xe nhỏ chỉ được tìm thấy khoảng 0.182, xe vừa 0.547, xe lớn 0.561. Nghĩa là xe ở xa bị bỏ sót nhiều hơn xe ở gần. Ở vòng 0, khung AI còn lệch ở một số xe bị che khuất hoặc nằm ở rìa. Tuy nhiên, nhãn dùng để chấm cũng do máy vẽ, chưa có người xem từng khung, nên có thể nhãn chấm sai chứ không phải AI của bạn sai.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Mỗi ảnh có một điểm. Một nửa điểm là AI không chắc. Ba phần mười là AI vẽ nhiều khung còn lưỡng lự. Hai phần mười là ảnh có khác thời gian với ảnh khác. Hai ảnh trong cùng lô phải cách nhau ít nhất 2 giây (vai trò của MIN_GAP_S), vì camera đứng yên, ảnh sát nhau gần như giống hệt. Rồi nhắc lại 3 ảnh frame_0182.jpg, frame_0099.jpg và frame_0107.jpg, và 1 ảnh bạn đã viết ở SELECTION.md là frame_0372.jpg. Điểm cao không có nghĩa sửa ảnh đó sẽ làm AI giỏi hơn.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 309 | 0.450 | -0.321 | 1.000 | 0.141 | 0.248 | 0.000 | 0.125 | 0.488 |

Trong vòng 1, số lượng box giữ nguyên là 156, chỉnh sửa 5, xoá 8, thêm mới 148. So điểm vòng 1 (0.450) với vòng 0 (0.771) cho thấy mô hình kém đi đáng kể. Nhóm xe nhỏ giảm xuống 0, nhóm xe lớn giảm xuống 0.488. Mắt tôi thấy trong BLIND_SCAN.md nhiều xe nhỏ bị sót, khung tôi sửa trên CVAT ở REVIEW_LOG.csv đã bù đắp phần đó, nhưng AI sau khi học lại có thể bị overfit vào lượng nhãn ít nên ra kết quả xấu đi. Một ca khó là xe ở sát mép khung bị cắt một phần.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

Điểm vòng 1 giảm mạnh so với vòng 0, chỉ còn 0.450 so với 0.771. Tôi chọn dừng vì sửa thêm thì mất thời gian, và không nên chọn hai ảnh sát nhau vì chúng gần như một cảnh. Hai chỗ còn yếu là xe xa chỉ còn hai chấm đèn, hoặc xe bị cắt mép. Phần chấm chỉ có 20 ảnh, xe quá nhỏ không tính, nhãn chấm chưa được người kiểm. Nếu điểm giảm, hãy xem lại khung bạn đã sửa trước khi cho AI học thêm.
