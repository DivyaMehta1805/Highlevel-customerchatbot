import { createRouter, createWebHistory } from 'vue-router'

import MainPage from './MainPage.vue';
import Chatbot from './Chatbot.vue';

const routes = [
  { path: '/', component: MainPage },
  { path: '/chatbot', component: Chatbot },
 
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router;