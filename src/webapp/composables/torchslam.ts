
export const useTorchSLAM = () => {

    return useState('landmark_locations', () => {

        const ws = new WebSocket('ws://localhost:7000');
        const landmark_locations = {xyz: [[]]};
        ws.onmessage = (event: any) => {
            console.log(event);
            event = JSON.parse(event.data);
            console.log('WebSocket message received:', event);
            if (event.type === 'all-landmark-locations') {
                landmark_locations.xyz = event.data.xyz;
            }
        };

        const request_event = {
            type: 'frontend-request',
            data: {
                topic: 'all-landmark-locations'
            }
        };
        ws.onopen = (event: any) => {
            console.log('WebSocket is open now.');
            ws.send(JSON.stringify(request_event));
        };

        return {
            landmark_locations
        }
    })
}