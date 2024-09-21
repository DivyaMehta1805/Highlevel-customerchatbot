// store/messages.js
import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useMessagesStore = defineStore('messages', () => {
  const messages = ref([]);

  function addMessage(message) {
    messages.value.push(message);
  }

  function updateMessage(id, updatedMessage) {
    const index = messages.value.findIndex((msg) => msg.id === id);
    if (index !== -1) {
      messages.value[index] = { ...messages.value[index], ...updatedMessage };
    }
  }

  return { messages, addMessage, updateMessage };
});
