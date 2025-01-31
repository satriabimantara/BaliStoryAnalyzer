function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth", // Use smooth scroll behavior
  });
}
// Show/hide scroll to top button based on scroll position
window.onscroll = function () {
  var scrollToTopBtn = document.getElementById("scrollToTopBtn");
  if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
    scrollToTopBtn.style.display = "block";
  } else {
    scrollToTopBtn.style.display = "none";
  }
};
