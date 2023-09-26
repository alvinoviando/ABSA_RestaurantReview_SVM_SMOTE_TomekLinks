$(window).scroll(function () {
  if ($(window).scrollTop() > 10) {
    $("#navbar").addClass("floatingNav");
  } else {
    $("#navbar").removeClass("floatingNav");
  }
});

//event
$(".page-scroll").on("click", function (e) {
  var tujuan = $(this).attr("href");
  var elemenTujuan = $(tujuan);

  $("html , body").animate(
    {
      scrollTop: elemenTujuan.offset().top - 90,
    },
    1000,
    "easeInOutExpo"
  );

  e.preventDefault();
});

$("#train-form").on("submit", function (e) {
  e.preventDefault();
  var file_train = new FormData($("#train-form")[0]);
  $(".load-icon-train").show();
  $.ajax({
    data: file_train,
    contentType: false,
    cache: false,
    processData: false,
    // async: false,
    type: "post",
    url: "/training",
  }).done(function (data) {
    $(".load-icon-train").hide();
    $("#nilai_akurasi_train").html(data.accuracy_train);
    $("#nilai_loss_train").html(data.loss_train);
    $("#nilai_akurasi_val").html(data.accuracy_val);
    $("#nilai_loss_val").html(data.loss_val);
    $("#text_input").html(data.text_input);
    $("#text_case_folding").html(data.text_case_folding);
    $("#text_stopword").html(data.text_stopword);
    $("#text_punct").html(data.text_punct);
    $("#text_tokenisasi").html(data.text_tokenisasi);
    $("#loss_plot").append(
      "<center><img src=" +
        data.loss_plot +
        " alt='Loss History Chart' width='400'></center>"
    );
    $("#acc_plot").append(
      "<center><img src=" +
        data.acc_plot +
        " alt='Accuracy History Chart' width='400'></center>"
    );
    $("#preproccessing").show();
    $("#hasil_training").show();
  });
});

$("#test-form").on("submit", function (e) {
  e.preventDefault();
  var file_test = new FormData($("#test-form")[0]);
  $(".load-icon-test").show();
  $.ajax({
    data: file_test,
    contentType: false,
    cache: false,
    processData: false,
    // async: false,
    type: "post",
    url: "/testing",
    // beforeSend: function() { $('.load-icon').hide();},
  }).done(function (data) {
    $(".load-icon-test").hide();
    $("#keterangan").show();
    //$('#nilai_akurasi').html(data.accuracy);
    $("#nilai_akurasi_test").append("Nilai Akurasi : " + data.accuracy);
    var table = $(".display").append(
      "<thead><tr><th>No</th><th>Title</th><th>Label Score</th><th>Prediction</th></tr></thead><tbody>"
    );
    $.each(data.data_output, function (a, b) {
      table.append(
        "<tr><td>" +
          b.id +
          "</td><td>" +
          b.title +
          "</td><td>" +
          b.label_score +
          "</td><td>" +
          b.prediction +
          "</td></tr>"
      );
    });
    $(".display").append(
      "</tbody><tfoot><tr><th>No</th><th>Title</th><th>Label Score</th><th>Prediction</th></tr></tfoot>"
    );
    $("#empTable").DataTable({
      scrollY: "450px",
      scrollX: true,
      scrollCollapse: true,
      fixedHeader: true,
      lengthMenu: [
        [10, 25, 50, -1],
        [10, 25, 50, "All"],
      ],
    });
  });
});

$("#pred-form").on("submit", function (e) {
  e.preventDefault();
  var data_pred = $("#data_pred").val();
  console.log(data_pred);
  $(".load-icon-pred").show();
  $.ajax({
    data: { data_pred: data_pred },
    type: "post",
    url: "/predict",
  }).done(function (data) {
    $(".load-icon-pred").hide();
    $("#hasil_pred").html("Hasil Prediksi : <b>" + data.hasil_pred + "</b>");
  });
});
