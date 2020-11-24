// $(window).scroll(function () {
//     if ($(window).scrollTop() >= 50) {
// $('.landing_card').css('background','red');
//     } else {
// $('.landing_card').css('background','transparent');
//     }
// });

$(document).ready(function() {
    $('.select2').select2({
    closeOnSelect: false
    });
});

$("#rec_button").click(function() {
    $('html, body').animate({
        scrollTop: $("#CityRecommender").offset().top
    }, 2000);
});

// $(document).ready(function () {
//     var height1 = $('.content').height()
//     var height2 = $('body').height()

//     if (height1 > height2) {
//         $('.sidebar').height(height1)
//     } else {
//         $('.sidebar').height(height2)
//     }
// });