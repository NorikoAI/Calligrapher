package MaterialUI.core

import react.RClass
import react.RProps

@JsModule("@material-ui/core/AppBar")
private external val AppBarModule: dynamic

interface AppBarProps: RProps {
    var color: String
    var position: String
}

val AppBar: RClass<AppBarProps> = AppBarModule.default