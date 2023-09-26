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
    $('#plot_eval_makanan').empty()
    $('#plot_eval_minuman').empty()
    $('#plot_eval_pelayanan').empty()
    $('#plot_eval_tempat').empty()
    $('#plot_eval_harga').empty()
    $('#cm_makanan').empty()
    $('#cm_minuman').empty()
    $('#cm_pelayanan').empty()
    $('#cm_tempat').empty()
    $('#cm_harga').empty()
    $("#text_input").html(data.text_input);
    $("#text_case_folding").html(data.text_case_folding);
    $("#text_stopword").html(data.text_stopword);
    $("#text_punct").html(data.text_punct);
    $("#text_tokenisasi").html(data.text_tokenisasi);
    $("#text_stemming").html(data.text_stemming);
    $("#text_c_slang").html(data.text_c_slang);
    $("#plot_eval_harga").append(
      "<center><img src=" +
        data.plot_eval_harga +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_eval_makanan").append(
      "<center><img src=" +
        data.plot_eval_makanan +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_eval_minuman").append(
      "<center><img src=" +
        data.plot_eval_minuman +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_eval_pelayanan").append(
      "<center><img src=" +
        data.plot_eval_pelayanan +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_eval_tempat").append(
      "<center><img src=" +
        data.plot_eval_tempat +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#cm_makanan").append(
      "<center><img src=" +
        data.cm_makanan +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#cm_minuman").append(
      "<center><img src=" +
        data.cm_minuman +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#cm_pelayanan").append(
      "<center><img src=" +
        data.cm_pelayanan +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#cm_tempat").append(
      "<center><img src=" +
        data.cm_tempat +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#cm_harga").append(
      "<center><img src=" +
        data.cm_harga +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#preproccessing").show();
    $("#card1").show();
    $("#card2").show();
    $(".load-icon-train").hide();
  });
});

$("#test-form").on("submit", function (e) {
  e.preventDefault();
  var nama_resto = $("#nama_resto").val();;
  console.log(nama_resto);
  $(".load-icon-test").show();
  $.ajax({
    data: {nama_resto:nama_resto} ,
    // async: false,
    type: "post",
    url: "/testing",
    // beforeSend: function() { $('.load-icon').hide();},
  }).done(function (data) {
    // $("#table_ulasan_input").dataTable().fnDestroy();
    // $('#table_content').empty()
    $('#jumlah_review').empty()
    $('#plot_input_makanan').empty()
    $('#plot_input_minuman').empty()
    $('#plot_input_pelayanan').empty()
    $('#plot_input_tempat').empty()
    $('#plot_input_harga').empty()
    $("#jumlah_review").html(data.jumlah_review);
    $("#ulasan_input").html(data.ulasan_input);
    $("#nama_resto_bold").html(nama_resto);
    


    $("#plot_input_makanan").append(
      "<center><img src=" +
        data.plot_input_makanan +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_input_minuman").append(
      "<center><img src=" +
        data.plot_input_minuman +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_input_pelayanan").append(
      "<center><img src=" +
        data.plot_input_pelayanan +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_input_tempat").append(
      "<center><img src=" +
        data.plot_input_tempat +
        " alt='Acc Chart' width='100%'></center>"
    );
    $("#plot_input_harga").append(
      "<center><img src=" +
        data.plot_input_harga +
        " alt='Acc Chart' width='100%'></center>"
    );
    $(".load-icon-test").hide();
    // var table_session = table_session.replaceAll("&#34;", '"'); // handle tanda petik 2 ", yang berubah menjadi &34; karena bug di session atau flasknya
    // var table_session = table_session.replaceAll("\\", " "); // handle escape character dalam data, jadi "\" dalam data akan dihapus
    // var table_session = JSON.parse(table_session);
    $("#hasil_input").show();

    var process_input = data.df_process_input;
    console.log('1');
    console.log(process_input);
    // var process_input = process_input.replaceAll("\\", " ");
    // var process_input = process_input.replaceAll("&#39;", '"');
    // console.log('2');
    // console.log(process_input);
    // var process_input = JSON.parse(process_input);
    // console.log('3');
    // console.log(process_input);
    
    $("#table_ulasan_input").DataTable({
      data: process_input,
      columns: [
        { title: "Ulasan" },
        { title: "Makanan" },
        { title: "Minuman" },
        { title: "Pelayanan" },
        { title: "Tempat" },
        { title: "Harga" }
      ],
      searching: false,
      ordering: false,
      info: false,
      bDestroy: true,
      pageLength: 5,
      lengthMenu: [
        [5, 10, 20, -1],
        [5, 10, 20, "Todos"],
      ],
    });

    // $("#table_content").append('#table_ulasan_input');
    

    
    
    
    // $("#keterangan").show();
    // //$('#nilai_akurasi').html(data.accuracy);
    // $("#nilai_akurasi_test").append("Nilai Akurasi : " + data.accuracy);
    // var table = $(".display").append(
    //   "<thead><tr><th>No</th><th>Title</th><th>Label Score</th><th>Prediction</th></tr></thead><tbody>"
    // );
    // $.each(data.data_output, function (a, b) {
    //   table.append(
    //     "<tr><td>" +
    //       b.id +
    //       "</td><td>" +
    //       b.title +
    //       "</td><td>" +
    //       b.label_score +
    //       "</td><td>" +
    //       b.prediction +
    //       "</td></tr>"
    //   );
    // });
    // $(".display").append(
    //   "</tbody><tfoot><tr><th>No</th><th>Title</th><th>Label Score</th><th>Prediction</th></tr></tfoot>"
    // );
    // $("#empTable").DataTable({
    //   scrollY: "450px",
    //   scrollX: true,
    //   scrollCollapse: true,
    //   fixedHeader: true,
    //   lengthMenu: [
    //     [10, 25, 50, -1],
    //     [10, 25, 50, "All"],
  //     ],
  //   });
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
  pass
  });
});
