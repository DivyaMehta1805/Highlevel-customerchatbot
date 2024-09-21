<script setup>

</script>

<template>
  <div class="h-screen">
    <router-view></router-view>
  </div>
</template>

<style scoped>
.mobile-size {
  height: 100vh;
  width: 100%;
  inset: 0;
}
@media (max-width: 768px) {
  .message-bubble {
    max-width: 80%;
  }
  .wave {
    height: 80px;
  }
}

@media (min-width: 768px) {
  /* .laptop-height {
    height: 75vh;
  } */
  /* .inset-off {
    left: auto;
    top: auto;
  } */

  .md\:bottom-28 {
    bottom: 7rem; /* Original bottom distance for laptops */
  }
  .md\:right-12 {
    right: 3rem; /* Position from the right for laptops */
  }

  .md\:w-96 {
    width: 24rem; /* Width for laptops */
  }
  .md\:rounded-xl {
    border-radius: 0.75rem; /* Rounded corners for desktops */
  }
}
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.5s ease, transform 0.5s ease;
}

.fade-slide-enter, 
.fade-slide-leave-to /* .fade-slide-leave-active in <2.1.8 */ {
  opacity: 0;
  transform: translateY(20px);
}
section {
  position: relative;
  width: 100%;

  height: 8px;
  transform: scaleY(-1); /* Invert the entire section (waves and animation) */
}

.wave {
  position: absolute;
  bottom: -43px;
  left: 0;
  width: 100%;
  height: 80px;
  background: url(../src/assets/wave.png);
  background-size: 1000px 100px;
  animation: animate 2s ease-in-out infinite;
  animation-delay: calc(var(--i) * 0.25s);
}

.wave#wave1 {
  z-index: 1000;
  opacity: 1;
  background-position-x: 300px;
}

.wave#wave2 {
  z-index: 999;
  opacity: 0.5;
  /* opacity:1; */
  background-position-x: 200px;
}

.wave#wave3 {
  z-index: 998;
  opacity: 0.2;
  /* opacity:1; */
  background-position-x: 100px;
}

.wave#wave4 {
  z-index: 997;
  opacity: 0.1;
  /* opacity:1; */
  background-position-x: 0px;
}

@keyframes animate {
  0%,
  100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px); /* Invert the animation movement */
  }
}

.mb-4 {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.message-bubble {
  max-width: 60%;
  padding: 8px 12px;
  border-radius: 12px;
  overflow-wrap: break-word;
  word-wrap: break-word;
  white-space: pre-wrap;
  margin-bottom: 8px;
  display: inline-block;
}

.message-bubble.user {
  background-color: #4a4a4a;
  align-self: flex-end;
}

.message-bubble.ai {
  background-color: #007bff;
  align-self: flex-start;
}

.message-bubble p {
  margin: 0;
  overflow-wrap: break-word;
  word-wrap: break-word;
}
::-webkit-scrollbar {
  width: 12px;
}

::-webkit-scrollbar-thumb {
  background: #374151;
  border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
  background: #555;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 6px;
}

.scrollbar-container {
  scrollbar-width: thin;
  scrollbar-color: #888 #f1f1f1;
}

.scrollable {
  max-height: 500px;
  overflow-y: auto;
}

.loader {
  width: 5px;
  aspect-ratio: 1;
  border-radius: 50%;
  animation: l5 1s infinite linear alternate;
}

@keyframes l5 {
  0% {
    box-shadow: 8px 0 #000, -8px 0 #0002;
    background: #000;
  }
  33% {
    box-shadow: 8px 0 #000, -8px 0 #0002;
    background: #0002;
  }
  66% {
    box-shadow: 8px 0 #0002, -8px 0 #000;
    background: #0002;
  }
  100% {
    box-shadow: 8px 0 #0002, -8px 0 #000;
    background: #000;
  }
}
</style>
