$(document).ready(function () {
    $.ajax({
        url: '/api/getUserCount',
        method: 'GET',
        success: function (response) {
            $('.count_object').text(response.count);
        },
        error: function (e) {
            console.error("Không thể lấy số lượng user.", e);
        }
    });

});
$(document).ready(async function() {

    // load user attendance
    current_count = 0;
    function fecthDataList() {
        $.ajax({
            url: '/get_users_attendance',
            method: 'GET',
            success: function(usersInfo) {
                // console.log(usersInfo.length)
                if (current_count != usersInfo.length && usersInfo.length) {
                    updateDataList(usersInfo)
                    current_count = usersInfo.count
                }
            },
            error: function(error) {
                console.error('Error fetching user info:', error);
            }
        });
    }
    fecthDataList()
    function updateDataList(usersInfo) {
        const $notificationDiv = $('#list_user_attendance');
    
        // Xóa nội dung hiện tại
        $notificationDiv.empty();
        
        // Tạo một box cho mỗi người dùng
        usersInfo.forEach(user => {
            const userBox = `
                <div class="user-box">
                    <p>${user.user_id}</p>
                    <p>${user.prediction_result}</p>
                    <p>${user.recorded_at}</p>
                    <p class="prediction-percent">${user.prediction_percent}%</p>
                </div>
            `;
            $notificationDiv.append(userBox);
        });
    }

    const videoElement = $('#webcam').get(0);
    let mediaRecorder;
    let recordedChunks = [];
    let stream;
    let start = true

    // Tự động mở webcam khi trang tải
    try {
        await startRecord();
    } catch (error) {
        console.error("Lỗi khi mở webcam:", error);
        return;
    }
    $('#recordBtn').on('click', async function() {
        await startRecord();
    });

    // Dừng quay video
    $('#stopRecordBtn').on('click', function() {
        if (mediaRecorder) {
            stopRecord();
        } else {
            console.error("MediaRecorder chưa được khởi tạo.");
        }
    });
    

    // Lật ngang (flip horizontal) khi nhấn nút "Flip Horizontal"
    $('#flipHorizontalBtn').on('click', function() {
        $('#webcam').toggleClass('flip-horizontal');
    });


    // Hàm bắt đầu quay video

    async function initStream () {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    }
    async function startRecord() {
        await initStream();
        // Khởi tạo mediaRecorder
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];
    
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) recordedChunks.push(event.data);
        };
    
        mediaRecorder.onstop = function() {
            // Dừng tất cả các track trong stream
            stream.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null; // Xóa nguồn video
            recordedChunks = [];  // Dọn dẹp dữ liệu đã ghi
        };
        start = true
    
        mediaRecorder.start();
        $('#recordBtn').prop('disabled', true); // Vô hiệu hóa nút "Bắt đầu"
        $('#stopRecordBtn').prop('disabled', false); // Kích hoạt nút "Kết thúc"
    }
    
    function stopRecord() {
        $('#recordBtn').prop('disabled', false);  // Đặt nút ghi khả dụng
        $('#stopRecordBtn').prop('disabled', true);
        
        // Chỉ gọi mediaRecorder.stop() mà không cần lặp lại trong onstop
        mediaRecorder.stop();
        start = false
        updatePredictionResult("Không xác định", "0%");
    }
    

    // bat tat cam khi chuyen tab
    $(document).on("visibilitychange", async function() {
        if (document.hidden) {
            stopRecord();  // Tắt camera khi chuyển khỏi tab
        } else {
            await startRecord(); // Bật lại camera khi quay lại tab
        }
    });

    // Kết quả dự đoán
    function updatePredictionResult(prediction, percent) {
        $('#resultText').text(prediction);
        $('#result_percentText').text(percent);
        $('#predictionResult').show();
    }


    // predict
    function predictImage() {
        const canvas = document.createElement('canvas');
        
        // Thiết lập kích thước lớn hơn cho độ phân giải cao hơn
        canvas.width = 400;  // Điều chỉnh kích thước này để ảnh rõ nét hơn
        canvas.height = 400;
    
        const context = canvas.getContext('2d');
        context.translate(canvas.width, 0);  // Di chuyển hệ trục tọa độ
        context.scale(-1, 1);  // Lật theo chiều ngang
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
        canvas.toBlob(function(blob) {
            const formData = new FormData();
            formData.append('image', blob, 'image.jpg');
    
            $.ajax({
                url: '/predict',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(data) {
                    $('#resultText').text(data.predicted_class);
                    $('#result_percentText').text(data.prediction_percent);
                    fecthDataList() // fetch list user attendanced
                },
                error: function(error) {
                    console.error('Error predicting image:', error);
                }
            });
        }, 'image/jpeg', 1);  // Thêm chất lượng JPEG cao hơn (0.95)
    }    
    
    // gọi hàm dự đoán liên tục
    setInterval(function(){
        if (start)
            predictImage()
    }, 500);

    function showLoader() {
        $('#btn_train_model').prop('disabled', true);
        $(".container_load").addClass("active");
        $(".container_form").hide();
    }

    // Ẩn preloader khi quá trình kết thúc
    function hideLoader() {
        $(".container_load").removeClass("active");
        $(".container_form").show();
    }

    const socket = io();  // Khởi tạo kết nối Socket.IO

    $('#btn_train_model').on('click', function() {
        showLoader();

        // Gọi route huấn luyện
        $.ajax({
            url: '/train_model',
            type: 'GET',
            contentType: false,
            processData: false,
            success: function() {
                // hideLoader();
                // window.location.href = '/';
            },
            error: function(error) {
                console.error('Error:', error);
                hideLoader();
            }
        });

        
    });
    // Lắng nghe sự kiện 'progress' từ server
    socket.on('progress', function(data) {
        showLoader()
        console.log(data.percent)
        const progressBar = $('#progress_bar');
        const loadText = $('#load_text');
        $('#progress_bar').attr('value', data.percent);
        loadText.text(`Loading ${data.percent}% ...`);

        if (data.percent >= 100) {
            hideLoader();
            setTimeout(() => {
                $('#btn_train_model').prop('disabled', false);
                window.location.href = '/';
            }, 1000);
        }
    });
    
    
});

