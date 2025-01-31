$(document).ready(function () {
  // Add scroll event listener
  $(window).scroll(function () {
    var scroll = $(window).scrollTop();
    var navbar = $("#navbar");

    // Add 'scrolled' class when scrolled down
    if (scroll > 0) {
      navbar.addClass("scrolled");
    } else {
      // Remove 'scrolled' class when back to top
      navbar.removeClass("scrolled");
    }
  });
});
