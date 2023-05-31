<!-- hide the footer from "cover" layout -->
<template>
    <footer
    v-if="$slidev.nav.currentLayout !== 'cover'"
    >
        <div class="absolute left-10px bottom-10px">
Research Affiliates
        </div>


        <div class="absolute right-30px bottom-10px">
Static Frame
        </div>

        <div class="absolute right-10px bottom-3px">
        <Transform :scale="0.6">
        <SlideCurrentNo />
        </Transform>
        </div>
    </footer>
  </template>
