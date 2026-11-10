import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

import org.streetpea.chiaking

import "controls" as C

// Protected zones of frame generation, grouped in per-game profiles. Used in the settings
// page and in the in-stream frame generation panel. Changes are saved and applied live.
ColumnLayout {
    id: editor
    property int selectedZone: -1
    readonly property var profiles: Chiaki.settings.frameGenerationZoneProfiles
    readonly property var zones: Chiaki.settings.frameGenerationZones
    spacing: 6

    onZonesChanged: {
        if (selectedZone >= zones.length)
            selectedZone = zones.length - 1;
    }

    RowLayout {
        spacing: 6
        Label { text: qsTr("Profile:") }

        C.ComboBox {
            id: profileCombo
            Layout.preferredWidth: 240
            model: editor.profiles
            currentIndex: Math.max(0, editor.profiles.indexOf(Chiaki.settings.frameGenerationZoneProfile))
            onActivated: (index) => {
                Chiaki.settings.frameGenerationZoneProfile = editor.profiles[index];
                editor.selectedZone = -1;
            }
        }

        C.Button {
            text: qsTr("New")
            Material.roundedScale: Material.SmallScale
            onClicked: {
                Chiaki.settings.addFrameGenerationZoneProfile(qsTr("Game"), false);
                editor.selectedZone = -1;
            }
        }

        C.Button {
            text: qsTr("Duplicate")
            Material.roundedScale: Material.SmallScale
            onClicked: Chiaki.settings.addFrameGenerationZoneProfile(Chiaki.settings.frameGenerationZoneProfile, true)
        }

        C.Button {
            text: qsTr("Delete")
            Material.roundedScale: Material.SmallScale
            enabled: editor.profiles.length > 1
            onClicked: {
                Chiaki.settings.removeFrameGenerationZoneProfile();
                editor.selectedZone = -1;
            }
        }
    }

    RowLayout {
        spacing: 6
        Label { text: qsTr("Name:") }

        TextField {
            id: nameField
            Layout.preferredWidth: 240
            text: Chiaki.settings.frameGenerationZoneProfile
            selectByMouse: true
            onEditingFinished: {
                if (text !== Chiaki.settings.frameGenerationZoneProfile)
                    Chiaki.settings.renameFrameGenerationZoneProfile(text);
                text = Qt.binding(() => Chiaki.settings.frameGenerationZoneProfile);
            }
        }

        Label {
            opacity: 0.7
            text: qsTr("(Enter to rename)")
        }
    }

    Label {
        visible: editor.zones.length === 0
        opacity: 0.7
        text: qsTr("No zones in this profile.")
    }

    Repeater {
        model: editor.zones

        RowLayout {
            id: zoneRow
            required property var modelData
            required property int index
            readonly property bool selected: editor.selectedZone === index
            spacing: 4

            function commit(key, percent) {
                const z = { x: modelData.x, y: modelData.y, w: modelData.w, h: modelData.h };
                z[key] = percent;
                Chiaki.settings.setFrameGenerationZone(index, z.x, z.y, z.w, z.h);
            }

            ToolButton {
                Layout.preferredWidth: 80
                text: qsTr("Zone %1").arg(zoneRow.index + 1)
                checkable: true
                checked: zoneRow.selected
                onClicked: editor.selectedZone = zoneRow.selected ? -1 : zoneRow.index
            }

            Repeater {
                model: [
                    { key: "x", label: qsTr("X") },
                    { key: "y", label: qsTr("Y") },
                    { key: "w", label: qsTr("W") },
                    { key: "h", label: qsTr("H") }
                ]

                RowLayout {
                    required property var modelData
                    spacing: 2
                    Label { text: modelData.label }
                    SpinBox {
                        Layout.preferredWidth: 120
                        from: 0
                        to: 1000
                        stepSize: 5
                        editable: true
                        value: Math.round(zoneRow.modelData[modelData.key] * 10)
                        textFromValue: (v, locale) => (v / 10).toFixed(1)
                        valueFromText: (text, locale) => Math.round(parseFloat(text.replace(",", ".")) * 10)
                        onActiveFocusChanged: {
                            if (activeFocus)
                                editor.selectedZone = zoneRow.index;
                        }
                        onValueModified: zoneRow.commit(modelData.key, value / 10)
                    }
                }
            }

            C.Button {
                text: qsTr("Remove")
                Material.roundedScale: Material.SmallScale
                onClicked: {
                    Chiaki.settings.removeFrameGenerationZone(zoneRow.index);
                    editor.selectedZone = -1;
                }
            }
        }
    }

    C.Button {
        text: qsTr("Add zone")
        Material.roundedScale: Material.SmallScale
        enabled: editor.zones.length < 16
        onClicked: {
            Chiaki.settings.addFrameGenerationZone();
            editor.selectedZone = editor.zones.length - 1;
        }
    }
}
