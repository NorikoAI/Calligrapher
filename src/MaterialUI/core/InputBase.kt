package MaterialUI.core

import react.RClass
import react.RProps

@JsModule("@material-ui/core/InputBase")
private external val InputBaseModule: dynamic

interface InputBaseProps: RProps {
    var classes: String
    var inputProps: dynamic
    var placeholder: String
}

val InputBase : RClass<InputBaseProps> = InputBaseModule.default