/** @type {import('tailwindcss').Config} */
export default {
  purge: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  content: [],
  theme: {
    extend: {
      colors:{
        'dark-blue':'#04243c',
        'yellow':'#f4bf04',
        'light-blue':'#2692f2',
        'green':'#49c628'
      },
      height:{
        '600px':'600px'
      },
      maxWidth:{
        '340px':'340px'
      },
      backgroundImage: {
        'wave-pattern': "url('../assets/wave_background.svg')",
      }
    },
  },
  plugins: [],
}


