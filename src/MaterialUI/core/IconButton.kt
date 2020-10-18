package MaterialUI.core

import react.RClass
import react.RProps

@JsModule("@material-ui/core/IconButton")
private external val IconButtonModule: dynamic

interface IconButtonProps: RProps {
    var className: String
    var color: String
    var edge: String
}

val IconButton : RClass<IconButtonProps> = IconButtonModule.default