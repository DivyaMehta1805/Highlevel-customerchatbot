<script setup>
import { ref, watch, nextTick } from "vue";
import axios from "axios";
import MessageIcon from "@mui/icons-material/Message";
import bot from "../src/assets/bot.png";
import wave from "./assets/wave.vue";
import darkblueWave from "./assets/darkbluewave.png";
import lightblueWave from "./assets/lightbluewave.png";
import greenWave from "./assets/greenwave.png";
import yellowWave from "./assets/yellowwave.png";
import defaultWave from "./assets/wave.png";
import blueWave from "./assets/bluewave.png";
import { useMessagesStore } from "./store/messages";

const { messages, addMessage, updateMessage } = useMessagesStore();

function handleChat() {
  isChatOpen.value = !isChatOpen.value;
}

const isChatOpen = ref(false);
const newMessage = ref("");
const fileName = ref(""); // Reactive variable for the file name

const selectedFile = ref(null); // For file upload
const isLoading = ref(false);
const chatContainer = ref(null);
const lastMessage = ref(null);
const scrollToBottom = () => {
  // Check if the lastMessage exists in the DOM before scrolling
  if (lastMessage.value && lastMessage.value.scrollIntoView) {
    lastMessage.value.scrollIntoView({ behavior: "smooth" });
  }
};

// Watch for changes in messages and scroll when they update
watch(messages, async () => {
  // Wait for the DOM to update before scrolling
  await nextTick();
  scrollToBottom();
});


const postMessage = async (userMessage) => {
  try {
    // Using JSONPlaceholder API to simulate interaction
    const response = await axios.post(
      "http://localhost:8003/query",
      null, {
  params: {
    query: userMessage,
    n_results: 1 // You can adjust this number as needed
  }
});

    // Extracting the response from JSONPlaceholder
    // return `Received response: ${response.data.title} - ${response.data.body}`;
  } catch (error) {
    console.error("Error with API:", error);
    return "I'm sorry, I couldn't process your request.";
  }
};
const getLLMResponse=async()=> {
  try {
    const response = await axios.get("http://localhost:8003/getresponse")
    return response.data.response;
  } catch (error) {
    console.error('Error getting LLM response:', error);
    this.error = 'An error occurred while fetching the response.';
  }
}


const sendMessage = async () => {
  if (newMessage.value.trim() === "" && !selectedFile.value) return;

  const formData = new FormData();
  let fileUploadResponse = null;

  if (selectedFile.value) {
    formData.append("file", selectedFile.value);
    formData.append("message", newMessage.value);
  }
  

  addMessage({
    id: Date.now(),
    text: newMessage.value,
    isAI: false,
  });

  try {
    if (selectedFile.value) {
      fileUploadResponse = await axios.post("https://file.io", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("File uploaded:", fileUploadResponse.data);
    }

    const loaderMessageId = Date.now() + 1;
    addMessage({
      id: loaderMessageId,
      text: "",
      isAI: true,
      isLoading: true,
    });

    await postMessage(newMessage.value);
    const getResponse=await getLLMResponse();

    updateMessage(loaderMessageId, {
      text: getResponse,
      isLoading: false,
    });
  } catch (error) {
    console.error("Error posting data:", error);
    updateMessage(loaderMessageId, {
      text: "Error generating response.",
      isLoading: false,
    });
  } finally {
    newMessage.value = "";
    selectedFile.value = null;
    removeFile();
  }
};

function getWaveStyle(waveNumber) {
  // if (waveNumber === 4) {
  //   return {
  //     backgroundImage: `url(${blueWave})`,
  //   };
  // }
  // if (waveNumber === 2) {
  //   return {
  //     backgroundImage: `url(${lightblueWave})`,
  //   };
  // }
  // if (waveNumber === 3) {
  //   return {
  //     backgroundImage: `url(${greenWave})`,
  //   };
  // }
  return {
    backgroundImage: `url(${defaultWave})`,
  };
}

// Handle file input change
const handleFileUpload = (event) => {
  const file = event.target.files[0];
  if (file) {
    selectedFile.value = file;

    fileName.value = file.name;
  }
};

// Remove the selected file
const removeFile = () => {
  selectedFile.value = null;
  newMessage.value = "";
  fileName.value = "";
};
</script>

<template>
  <div class="relative h-screen">
    <transition name="fade-slide">
      <div
        class="fixed mobile-size inset-off laptop-height md:rounded-xl bg-dark-blue text-white md:overflow-hidden"
      >
        <section>
          <div
            class="wave"
            id="wave1"
            :style="getWaveStyle(1)"
            style="--i: 1"
          ></div>
          <div
            class="wave"
            id="wave2"
            :style="getWaveStyle(2)"
            style="--i: 2"
          ></div>
          <div
            class="wave"
            id="wave3"
            :style="getWaveStyle(3)"
            style="--i: 3"
          ></div>
          <div
            class="wave"
            id="wave4"
            :style="getWaveStyle(4)"
            style="--i: 4"
          ></div>
        </section>
        <div class="mt-10 flex justify-between items-center gap-2 px-6">
          <div class="flex gap-2">
            <img :src="bot" class="h-10 w-auto" />
            <div class="self-center">Bop</div>
          </div>

      
        </div>
        <div
          class="px-6 py-4 h-full mx-auto overflow-y-auto scrollable max-w-sm md:max-w-md lg:max-w-3xl flex flex-col justify-start"
          ref="chatContainer"
          style="height: calc(100% - 200px)"
        >
          <div
            v-for="message in messages"
            :key="message.id"
            class="mb-4 flex items-start space-x-2"
          >
            <!-- User Message -->
            <div
              v-if="!message.isAI"
              class="message-bubble user bg-blue-100 p-2 rounded-lg max-w-xs"
            >
              <p class="text-left text-sm">{{ message.text }}</p>
              <div v-if="message.fileName" class="text-xs mt-1">
                File: {{ message.fileName }}
              </div>
            </div>

            <!-- AI Message -->
            <div
              v-if="message.isAI"
              class="message-bubble ai bg-gray-200 p-2 rounded-lg max-w-xs"
            >
              <div v-if="message.isLoading" class="loader"></div>
              <p v-else class="text-left text-sm">{{ message.text }}</p>
            </div>
          </div>
          <div ref="lastMessage"></div>
        </div>

        <!-- Input Section -->
        <div
          class="absolute bottom-4 md:bottom-8 left-0 right-0 mx-auto p-2 md:p-4 bg-gray-800 rounded-lg flex flex-col space-y-1 md:space-y-3 max-w-340px md:max-w-md lg:max-w-3xl"
        >
          <!-- File Name and Remove Button -->
        

          <!-- File Upload and Message Input -->
          <div class="flex w-full items-center space-x-2 md:space-x-4">
            <!-- File Upload -->
            <!-- <div class="relative">
              <label
                class="flex items-center justify-center gap-2 w-12 h-12 bg-gray-700 text-white rounded-lg cursor-pointer hover:bg-gray-600 transition-all duration-200"
              >
                <i class="fas fa-upload text-xl text-blue-400"></i>
                <input
                  type="file"
                  @change="handleFileUpload"
                  class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
              </label>
            </div> -->

            <!-- Message Input -->
            <form @submit.prevent="sendMessage" class="relative flex-grow">
              <input
                v-model="newMessage"
                type="text"
                placeholder="Message or select a file..."
                class="w-full p-3 bg-gray-700 text-white outline-none rounded-lg shadow-lg focus:ring-2 focus:ring-blue-500 transition-all duration-200"
              />
            </form>
          </div>
        </div>
      </div>
    </transition>
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
