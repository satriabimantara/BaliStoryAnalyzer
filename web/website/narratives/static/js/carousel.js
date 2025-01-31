// Get the current vertical scroll position of the window
var currentPosition = window.scrollY || window.pageYOffset;

// Define the target scroll position (e.g., 1980)
var targetPosition = 1100;

// Calculate the distance to scroll
var distance = targetPosition - currentPosition;

// Define the duration of the scroll animation (in milliseconds)
var duration = 3000; // 1 second

// Define the start time of the animation
var startTime = performance.now();

// Function to perform smooth scroll animation
function smoothScroll() {
  // Calculate the elapsed time since the start of the animation
  var currentTime = performance.now();
  var elapsedTime = currentTime - startTime;

  // Calculate the next scroll position based on easing function (e.g., quadratic)
  var nextPosition =
    currentPosition + easeInOutQuad(elapsedTime, 0, distance, duration);

  // Scroll to the next position
  window.scrollTo(0, nextPosition);

  // Check if the animation is still in progress
  if (elapsedTime < duration) {
    // Continue the animation
    requestAnimationFrame(smoothScroll);
  }
}

// Easing function (quadratic easing in/out)
function easeInOutQuad(t, b, c, d) {
  t /= d / 2;
  if (t < 1) return (c / 2) * t * t + b;
  t--;
  return (-c / 2) * (t * (t - 2) - 1) + b;
}
