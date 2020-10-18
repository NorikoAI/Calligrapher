package MaterialUI.core

import react.RClass
import react.RProps

@JsModule("@material-ui/core/Toolbar")
private external val ToolbarModule: dynamic

interface ToolbarProps: RProps{
    var variant: String
}

val Toolbar : RClass<ToolbarProps> = ToolbarModule.default