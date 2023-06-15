<!-- hide the footer from "cover" layout -->
<template>
    <footer
    v-if="$slidev.nav.currentLayout !== 'none'"
    >
        <div class="absolute left-20px bottom-20px">
        <img src="/RA-Logo-White-TM.svg" class="h-8 opacity-26" />
        </div>

        <div class="absolute right-36px bottom-16px">
        <img src="/sf-logo-horiz.svg" class="h-9 opacity-26" />
        </div>
    </footer>

    <footer
    v-if="$slidev.nav.currentLayout !== 'cover'"
    >
        <div class="absolute right-10px bottom-3px">
        <Transform :scale="0.4">
        <SlideCurrentNo class="opacity-20"/>
        </Transform>
        </div>

    </footer>

  </template>
