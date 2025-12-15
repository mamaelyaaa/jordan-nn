import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { aliases, mdi } from 'vuetify/iconsets/mdi-svg'
import 'vuetify/styles'

import {
  mdiPlay,
  mdiStop,
  mdiPause,
} from '@mdi/js'

export default createVuetify({
  components,
  directives,

  icons: {
    defaultSet: 'mdi',
    aliases: {
      ...aliases,

      play: mdiPlay,
      stop: mdiStop,
      pause: mdiPause,
    },
    sets: {
      mdi,
    },
  },

  theme: {
    defaultTheme: 'light',
  },
})
