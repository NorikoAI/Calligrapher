plugins {
    id("org.jetbrains.kotlin.js") version "1.4.10"
}

group = "package org.sourcekey.NorikoAI.Calligrapher"
version = "1.0.0"

repositories {
    maven("https://kotlin.bintray.com/kotlin-js-wrappers/")
    mavenCentral()
    jcenter()
}

dependencies {
    implementation(kotlin("stdlib-js"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.9")
    //React, React DOM + Wrappers (chapter 3)
    implementation("org.jetbrains:kotlin-react:16.13.1-pre.110-kotlin-1.4.0")
    implementation("org.jetbrains:kotlin-react-dom:16.13.1-pre.110-kotlin-1.4.0")
    implementation(npm("react", "16.13.1"))
    implementation(npm("react-dom", "16.13.1"))
    //implementation(npm("react-is", "16.13.1"))
    //Video Player Share Buttons
    implementation(npm("react-player", "~2.6.0"))
    implementation(npm("react-share", "~4.2.1"))
    //Kotlin Styled (chapter 3)
    implementation("org.jetbrains:kotlin-css:1.0.0-pre.110-kotlin-1.4.0")
    implementation("org.jetbrains:kotlin-styled:1.0.0-pre.110-kotlin-1.4.0")
    implementation(npm("styled-components", "4.4.0"))
    implementation(npm("inline-style-prefixer", "6.0.0"))
    //MaterialUI Muirwik
    implementation(npm("@material-ui/core", "^4.11.0"))
    implementation(npm("@material-ui/icons", "^4.9.1"))
    implementation("com.ccfraser.muirwik:muirwik-components:0.6.0")
    //
    implementation(npm("@egjs/react-infinitegrid", "~3.0.5"))
    //JQuery
    implementation(npm("jquery", "3.5.1"))
    //Opentype.js
    implementation(npm("opentype.js", "~1.3.2"))
    //TensorFlow
    implementation(npm("@tensorflow/tfjs", "~2.4.0", generateExternals = false))
    implementation(npm("@tensorflow/tfjs-vis", "1.4.3", generateExternals = false))
}

kotlin {
    js {
        browser {
            webpackTask {
                cssSupport.enabled = true
            }

            runTask {
                cssSupport.enabled = true
            }

            testTask {
                useKarma {
                    useChromeHeadless()
                    webpackConfig.cssSupport.enabled = true
                }
            }
        }
        binaries.executable()

        useCommonJs()
    }
}