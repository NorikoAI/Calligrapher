package MaterialUI.core

import react.RClass
import react.RProps

@JsModule("@material-ui/core/Typography")
private external val TypographyModule: dynamic

interface TypographyProps: RProps {
    var className: String
    var variant: String
}

val Typography : RClass<TypographyProps> = TypographyModule.default