<script setup>
import { ref } from 'vue'
import bot from "./assets/bot.png"
import router from "./router";

// Reactive references
const url = ref('');
const loading = ref(false); // To track loading state

// Send the URL to the API and handle the response
const sendMessage = async () => {
  loading.value = true; // Show loader
  try {
    const response = await fetch('http://0.0.0.0:8007/crawl', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url.value }),
    });

    if (!response.ok) {
      router.push('/chatbot');
      throw new Error('Failed to send the URL');

    }

    const data = await response.json();
    console.log('Success:', data);
    if(response.ok)
    router.push('/chatbot');
    
  } catch (error) {
    console.error('Error:', error);
  } finally {
    loading.value = false; // Hide loader when done
  }
};
</script>


<template>
  <div class="bg-gradient-to-r from-gray-500 to-blue-900 justify-center w-full h-screen flex flex-col md:flex-row">
    
    <!-- Left Side: Logo and Welcome Message -->
    <div class="flex flex-col justify-center items-center w-full lg:w-1/2 p-6 lg:pl-16">
      <img :src="bot" class="h-32 w-auto mb-4 md:h-40" /> <!-- Adjusted size for mobile and desktop -->
      <h1 class="text-3xl md:text-5xl font-bold text-white mb-2 md:mb-4">Welcome to Bop!</h1> <!-- Smaller on mobile -->
      <p class="text-lg md:text-xl font-light text-white text-center md:text-left">Please enter the URL to get started.</p>
    </div>

    <!-- Right Side: URL Input -->
    <div class="flex flex-col justify-center items-center w-full lg:w-1/2 p-6 lg:pr-16">
      <div class="w-full max-w-xl">
        <form @submit.prevent="sendMessage">
          <input
            v-model="url"
            type="text"
            placeholder="Enter the URL"
            class="w-full p-4 md:p-5 bg-gray-700 text-white outline-none rounded-lg shadow-lg focus:ring-2 focus:ring-blue-500 transition-all duration-200"
          />
        </form>
      </div>

      <!-- Loader: Only visible when loading is true -->
      <div v-if="loading" class="flex justify-center mt-10">
        <span class="loader"></span>
      </div>
    </div>

  </div>
</template>


<style scoped>
.loader {
    width: 48px;
    height: 48px;
    border: 5px solid #FFF;
    border-bottom-color: #04243c;
    border-radius: 50%;
    display: inline-block;
    box-sizing: border-box;
    animation: rotation 1s linear infinite;
    }

    @keyframes rotation {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
    } 
</style>

