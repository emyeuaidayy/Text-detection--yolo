# Text-detection--yolo
Trong báo cáo này, tôi sẽ giới thiệu về phát hiện chữ trong hình ảnh, một lĩnh vực quan trọng trong xử lý ảnh và trí tuệ nhân tạo. Phát hiện chữ trong hình ảnh là quá trình tự động xác định vị trí và ranh giới của các vùng chứa chữ trong hình ảnh.
Mục tiêu của phát hiện chữ trong hình ảnh là xác định các vùng chứa chữ trong hình ảnh một cách chính xác và cung cấp thông tin về vị trí và kích thước của các vùng này. Quá trình này thường được thực hiện bằng cách sử dụng các thuật toán xử lý ảnh và mô hình máy học để phân tích các đặc trưng của chữ và tìm kiếm các vùng chứa chữ trong hình ảnh.
Sau khi đã phát hiện được các vùng chứa chữ, các vùng này có thể được chuyển đến bước nhận dạng chữ (OCR) để nhận dạng nội dung của chúng. Phát hiện chữ trong hình ảnh là một công đoạn quan trọng trong quy trình tổng thể của việc xử lý văn bản trong hình ảnh và có ứng dụng rộng rãi trong việc quét tài liệu, nhận dạng biển số xe, công nghệ OCR, và nhiều lĩnh vực khác.


1.	Mạng Yolo
a.	Tổng quan
YOLO (You Only Look Once) là một thuật toán phát hiện đối tượng trong lĩnh vực thị giác máy tính. Thuật toán YOLO đưa ra một cách tiếp cận mới và hiệu quả để đồng thời dự đoán các hộp giới hạn và xác suất lớp cho các đối tượng trong ảnh.
Truyền thống, các phương pháp phát hiện đối tượng phổ biến thường chia quá trình này thành hai giai đoạn: tìm kiếm các vùng quan tâm (region proposals) và sau đó thực hiện phân loại trên từng vùng đó. Tuy nhiên, YOLO đã đưa ra một cách tiếp cận khác, thực hiện cả hai công việc này trong một lần duy nhất.
Mạng YOLO được xây dựng dựa trên kiến trúc mạng neural convolutional và sử dụng một lớp fully connected cuối cùng để tạo ra các dự đoán. Thay vì xem xét ảnh điểm ảnh theo các vùng hoặc cửa sổ trượt như các phương pháp khác, YOLO chia ảnh thành một lưới ô lưới và dự đoán các hộp giới hạn và xác suất lớp cho mỗi ô lưới đó.
Điểm mạnh của YOLO là khả năng dự đoán nhanh và chính xác. Bằng cách thực hiện toàn bộ quá trình dự đoán trong một lần, YOLO có thể đạt được tốc độ cao và phù hợp cho các ứng dụng thời gian thực. Ngoài ra, YOLO cũng có khả năng xử lý các đối tượng nhỏ và đa dạng trong cùng một khung hình, mang lại hiệu suất ấn tượng.
!
b.	Cách Yolo hoạt động 
Với đầu vào là một hình ảnh, mô hình YOLO sẽ tạo ra một ma trận kết quả có kích thước S×S×(5×N+M). Trong đó, N là số lượng bounding box cần dự đoán trong mỗi ô và M là số lượng lớp đối tượng cần phân loại.
Ví dụ, nếu chúng ta chia hình ảnh thành một lưới ô kích thước 7×7 và mỗi ô cần dự đoán 2 bounding box và 3 đối tượng: con chó, ô tô, xe đạp, thì kết quả đầu ra sẽ là một ma trận kích thước 7×7×13. Mỗi ô sẽ có 13 tham số và tổng cộng có 98 bounding box được trả về.
Số lượng tham số (5×N+M) được tính bằng cách dự đoán mỗi bounding box gồm 5 thành phần: (x, y, w, h, prediction). Trong đó, (x, y) là tọa độ tâm của bounding box, (w, h) là chiều rộng và chiều cao của bounding box, và prediction là dự đoán được tính theo công thức (Pr(Object) * IOU(pred, truth)). Với hình ảnh ví dụ trên, mỗi ô sẽ có 13 tham số. Đơn giản nhìn, tham số thứ nhất (P(Object)) sẽ chỉ ra xem ô đó có chứa đối tượng hay không. Các tham số tiếp theo sẽ trả về thông tin về các bounding box. Tham số cuối cùng trong mỗi ô sẽ là xác suất của từng lớp đối tượng (ví dụ: P(chó|object), P(ô tô|object), P(xe đạp|object))
Lưu ý rằng tâm của bounding box sẽ nằm trong ô nào thì ô đó sẽ chứa đối tượng, dù đối tượng có thể nằm trên các ô khác. Điều này có nghĩa là nếu một ô chứa nhiều tâm bounding box hoặc đối tượng, mô hình sẽ không thể phát hiện được. Đây là một hạn chế của mô hình YOLO1, vì vậy chúng ta cần tăng số lượng ô chia trong một ảnh để cải thiện khả năng phát hiện đối tượng.

 

!


c.	Hàm tính IOU
Trên ta có đề cập prediction được định nghĩa Pr( Object )*IOU( pred, truth ), ta sẽ làm rõ hơn IOU(pred, truth) là gì. IOU (INTERSECTION OVER UNION) là hàm đánh giá độ chính xác của object detector trên tập dữ liệu cụ thể. IOU được tính bằng:
 
Trong đó Area of Overlap là diện tích phần giao nhau giữa predicted bounding box với grouth-truth bouding box , còn Area of Union là diện tích phần hợp giữa predicted bounding box với grouth-truth bounding box. Những bounding box được đánh nhãn bằng tay trong tập traing set và test set. Nếu IOU > 0.5 thì prediction được đánh giá là tốt.

!
d.	Hàm mất mát loss
Hàm lỗi trong YOLO được tính trên việc dự đoán và nhãn mô hình để tính. Cụ thể hơn nó là tổng độ lôĩ của 3 thành phần con sau :
-	Độ lỗi của việc dự đoán loại nhãn của object - Classifycation loss
-	Độ lỗi của dự đoán tọa độ tâm, chiều dài, rộng của boundary box (x, y ,w, h) - Localization loss
-	Độ lỗi của việc dự đoán bounding box đó chứa object so với nhãn thực tế tại ô vuông đó - Confidence loss

Classifycation loss
Classifycation loss - độ lỗi của việc dự đoán loại nhãn cuả object, hàm lỗi này chỉ tính trên những ô vuông có xuất hiện object, còn những ô vuông khác ta không quan tâm. Classifycation loss được tính bằng công thức sau:

!
 
-	Localization loss là hàm lỗi dùng để tính giá trị lỗi cho boundary box được dự đoán bao gồm tọa độ tâm, chiều rộng, chiều cao của so với vị trí thực tế từ dữ liệu huấn luyện của mô hình. Lưu ý rằng chúng ta không nên tính giá trị hàm lỗi này trực tiếp từ kích thức ảnh thực tế mà cần phải chuẩn hóa về [0, 1] so với tâm của bounding box. Việc chuẩn hóa này kích thước này giúp cho mô hình dự đoán nhanh hơn và chính xác hơn so với để giá trị mặc định của ảnh. Hãy cùng xem một ví dụ: 
-	Giá trị hàm Localization loss được tính trên tổng giá trị lỗi dự đoán toạ độ tâm (x, y) và (w, h) của predicted bounding box với grouth-truth bounding box. Tại mỗi ô có chưa object, ta chọn 1 boundary box có IOU (Intersect over union) tốt nhất, rồi sau đó tính độ lỗi theo các boundary box này.
Giá trị hàm lỗi dự đoán tọa độ tâm (x, y) của predicted bounding box và (x̂, ŷ) là tọa độ tâm của truth bounding box được tính như sau :

!
 
-	Giá trị hàm lỗi dự đoán (w, h ) của predicted bounding box so với truth bounding box được tính như sau :

!
 

-	Với ví dụ trên thì S =7, B =2, còn λcoord là trọng số thành phần trong paper gốc tác giả lấy giá trị là 5
Confidence loss
Confidence loss là độ lỗi giữa dự đoán boundary box đó chứa object so với nhãn thực tế tại ô vuông đó. Độ lỗi này tính trên cả những ô vuông chứa object và không chứa object.

!

Với ví dụ trên thì S =7, B =2, còn λnoobject là trọng số thành phần trong paper gốc tác giả lấy giá trị là 0.5. Đối với các hộp j của ô thứ i nếu xuất hiệu object thì Ci =1 và ngược lại
Total loss
Tổng lại chúng ta có hàm lỗi là tổng của 3 hàm lỗi trên 
 
!


2.	Phương pháp thực hiện 
Trong phần trên, chúng ta đã được giới thiệu qua những cơ sở lý thuyết cần có để thực hiện kiến trúc mạng Unet. Qua đó chúng ta có thể áp dụng mô hình trên để thực phát hiện chữ trên thẻ nhận dạng (ID card) trong một hình ảnh
a.	Chuẩn bị dữ liệu: 
Thu thập và chuẩn bị tập dữ liệu chứa hình ảnh các thẻ chứa chữ cần nhận diện.
Dữ liệu đầu vào (input)
-	Là một tập data bao gồm 1000 hình ảnh có chứa các hình ảnh có chứa chữ cái và các nhãn của chúng 
Tập dữ liệu Input : https://drive.google.com/drive/folders/1TjSjUzSWH0eMSlArVOx8fW2KxOVzoY87?usp=drive_link
Tập dữ liệu nhãn : 
https://drive.google.com/drive/folders/1iCXYo7kOINzrmmyM7SC53u8ZAp-HDMst?usp=drive_link

File dữ liệu json :
https://drive.google.com/file/d/182RH1Y-a3V80nVvpsDzzI_--437dcrz3/view?usp=sharingFile dữ liệu h5 : 
File dữ liệu h5 : 
https://drive.google.com/file/d/1YNNdujDVXjnoyGR-5Rdn1osaiKpjvtT7/view?usp=sharing
b.	Chuẩn hóa dữ liệu 
Chuẩn hóa dữ liệu hình ảnh bằng cách chia cho 127.5 và trừ đi 1 được gọi là chuẩn hóa min-max, một phương pháp thường được sử dụng để đưa dữ liệu về khoảng giá trị từ -1 đến 1. Phương pháp này có các bước sau:
Chia mỗi giá trị trong dữ liệu cho 127.5: Bằng cách chia cho giá trị này, chúng ta đảm bảo rằng giá trị lớn nhất trong dữ liệu sau khi chuẩn hóa là 1.
Trừ đi 1 từ dữ liệu đã được chia: Bằng cách trừ đi 1, chúng ta đưa giá trị nhỏ nhất trong dữ liệu về -1.
Kết quả sau quá trình chuẩn hóa là dữ liệu nằm trong khoảng giá trị từ -1 đến 1, điều này rất thuận tiện khi đưa dữ liệu vào mô hình huấn luyện. Phương pháp chuẩn hóa min-max này đảm bảo rằng dữ liệu được điều chỉnh sao cho phù hợp với đầu vào của mô hình và giúp cải thiện quá trình huấn luyện.
 
c.	Kiến trúc Yolo áp dụng cho mô hình 
Link hình ảnh  https://drive.google.com/file/d/1-58w_jiKbBxes4vsmvEatEtqesPukstl/view?usp=drive_link
d.	Kết quả thực nghiệm 
Mô hình nhận diện các tập trong dataset
 
Mô hình nhận diện thẻ ID card 
-	Vì chưa được huấn luyện cho tiếng việt nên một số từ tiếng việt mô hình sẽ không thể nhận diện ra được 
 

 
 
 
Link mô hình : 
https://colab.research.google.com/drive/12UwHXP1xLtq5tKzXxu_JTfEylrJCbBl5?usp=sharing
Tài liệu tham khảo :
-	Neerajj9/Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow: Implemented the YOLO algorithm for scene text detection in keras-tensorflow (No object detection API used) The code can be tweaked to train for a different object detection task using YOLO. (github.com)
-	Tìm hiểu về YOLO trong bài toán real-time object detection (viblo.asia)
