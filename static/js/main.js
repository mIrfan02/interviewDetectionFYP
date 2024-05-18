jQuery(function ($) {
  "use strict";

  /* Window Load ---------------------- */

  $(window).on("load", function () {});

  /* Document Ready ------------------- */

  $(document).ready(function () {
    AOS.init({
      duration: 2000,
    });
  });

  /* Window Scroll -------------------- */

  $(window).on("scroll", function () {
    let scroll = $(window).scrollTop();
    if (scroll >= 100) {
      $(".header-wrap").addClass("scrolled");
    } else {
      $(".header-wrap").removeClass("scrolled");
    }
  });

  /* Window Resize -------------------- */

  $(window).on("resize", function () {});
});

const typeText = () => {
  const letters = document.querySelectorAll(".letter");
  let delay = 0;
  const increment = 100;

  letters.forEach((letter) => {
    setTimeout(() => {
      letter.style.opacity = "1";
    }, delay);
    delay += increment;
  });

  setTimeout(() => {
    // Reset after all letters are typed out
    letters.forEach((letter) => {
      letter.style.opacity = "0";
    });
    setTimeout(typeText, 1000); // 1s delay before retyping
  }, delay + letters.length * increment);
};

document.addEventListener("DOMContentLoaded", typeText);
